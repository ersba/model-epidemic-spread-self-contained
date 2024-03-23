using System;
using System.Collections.Generic;
using Mars.Interfaces.Agents;
using Mars.Interfaces.Annotations;
using Mars.Interfaces.Layers;
using MathNet.Numerics.Distributions;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadSelfContained.Model 
{
    public class Host : IAgent<InfectionLayer>
    {
        [PropertyDescription]
        public int Index { get; set; }
        
        [PropertyDescription]
        public int MyAgeGroup { get; set; }
        
        [PropertyDescription]
        public int MyStage { get; set; }

        private Tensor _tensorMyStage;
        
        private LearnableParams _learnableParams;

        private int _infectedTime;
        
        private int _meanInteractions;
        
        private int _nextStageTime;

        private int _infinityTime;

        private bool _exposedToday;

        private float _susceptibility;
        
        [PropertyDescription] 
        public UnregisterAgent UnregisterHandle { get; set; }
        
        private static float[] _lamdaGammaIntegrals;

        public void Init(InfectionLayer layer)
        {
            _infectionLayer = layer;
            _infectionLayer.ContactEnvironment.Insert(this);
            _infinityTime = Params.Steps + 1;
            _learnableParams = LearnableParams.Instance;
            InitStage();
            InitTimeVariables();
            InitMeanInteractions();
            _susceptibility = Params.Susceptibility[MyAgeGroup];
        }

        public void Tick()
        {
            // RecoveredOrDead = _tensorMyStage * (MyStage == (int)Stage.Infected) * (_nextStageTime <= _infectionLayer.GetCurrentTick()) / (float)Stage.Infected;
            // _infectionLayer.Insert(Index, RecoveredOrDead);
            _infectionLayer.Insert(Index, _tensorMyStage, tf.constant(_nextStageTime));
            _exposedToday = false;
            Interact();
            Progress();
            if (MyStage == (int)Stage.Mortality) Die();
        }
        private void Interact()
        {
            if(MyStage == (int) Stage.Susceptible){
                foreach (Host host in _infectionLayer.ContactEnvironment.GetNeighbors(Index))
                {
                    if (host.MyStage is (int) Stage.Infected or (int) Stage.Exposed)
                    {
                        var infector = Params.Infector[host.MyStage];
                        var bN = Params.EdgeAttribute;
                        var integral =
                            _lamdaGammaIntegrals[
                                Math.Abs(_infectionLayer.GetCurrentTick() - host._infectedTime)];
                        var result = Params.R0Value * _susceptibility * infector * bN * integral / _meanInteractions;

                        Random random = new Random();
                        if (!(random.NextDouble() < result)) continue;
                        _exposedToday = true;
                        return;
                    }
                }
            }
        }
        
        private void InitStage()
        {
            var p = tf.stack(new[] {_learnableParams.InitialInfectionRate, 1 - _learnableParams.InitialInfectionRate});
            _tensorMyStage = tf.cast(GumbelSoftmax.Execute(p)[0], dtype: TF_DataType.TF_FLOAT) * 2; // Infected = 2
            _tensorMyStage = tf.squeeze(_tensorMyStage);
            MyStage = (int) tf.cast(_tensorMyStage, TF_DataType.TF_INT32);
        }

        private void InitTimeVariables()
        {
            switch (MyStage)
            {   
                case (int) Stage.Susceptible:
                    _infectedTime = _infinityTime;
                    _nextStageTime = _infinityTime;
                    break;
                case (int)Stage.Exposed:
                    _infectedTime = 0;
                    _nextStageTime = Params.Steps + Params.ExposedToInfectedTime;
                    break;
                case (int)Stage.Infected:
                    _infectedTime = 1 - Params.ExposedToInfectedTime;
                    _nextStageTime = 1 + Params.InfectedToRecoveredTime;
                    break;
                case (int)Stage.Recovered:
                    _infectedTime = _infinityTime;
                    _nextStageTime = _infinityTime;
                    break;
                case (int)Stage.Mortality:
                    _infectedTime = _infinityTime;
                    _nextStageTime = _infinityTime;
                    break;
            }
        }
        
        private void InitMeanInteractions()
        {
            var childAgent = MyAgeGroup <= Params.ChildUpperIndex;
            var adultAgent = MyAgeGroup > Params.ChildUpperIndex && MyAgeGroup <= Params.AdultUpperIndex;
            var elderAgent = MyAgeGroup > Params.AdultUpperIndex;
            
            if (childAgent) _meanInteractions = Params.Mu[0];
            else if (adultAgent) _meanInteractions = Params.Mu[1];
            else if (elderAgent) _meanInteractions = Params.Mu[2];
        }

        private void Die()
        {
           // UnregisterHandle.Invoke(_infectionLayer, this);
        }

        private void Progress()
        {
            var nextStage = UpdateStage();
            UpdateNextStageTime();
            MyStage = nextStage;
            if (_exposedToday) _infectedTime = (int)_infectionLayer.GetCurrentTick();
        }
        
        private void UpdateNextStageTime()
        {
            if (_exposedToday)
            {
                _nextStageTime = (int) (_infectionLayer.GetCurrentTick() + 1
                                                                         + Params.ExposedToInfectedTime);
            } else if (_nextStageTime == _infectionLayer.GetCurrentTick())
            {
                if (MyStage == (int)Stage.Exposed)
                    _nextStageTime = (int) (_infectionLayer.GetCurrentTick()
                                            + Params.InfectedToRecoveredTime);
                else
                    _nextStageTime = _infinityTime;
            }
        }

        private int UpdateStage()
        {
            if (_exposedToday)
            {
                _tensorMyStage = tf.constant((float)Stage.Exposed);
                return (int)Stage.Exposed;
            }
            switch (MyStage)
            {
                case (int)Stage.Susceptible:
                    return (int)Stage.Susceptible;
                case (int)Stage.Exposed:
                    if (_infectionLayer.GetCurrentTick() >= _nextStageTime)
                    {
                        _tensorMyStage = tf.constant((float)Stage.Infected) * _tensorMyStage / (float)Stage.Exposed; 
                        return (int)Stage.Infected;
                    }
                    return (int)Stage.Exposed;
                case (int)Stage.Infected:
                    if (_infectionLayer.GetCurrentTick() >= _nextStageTime)
                    {
                        Random random = new Random();
                        if (random.NextDouble() < (double) tf.cast(_learnableParams.MortalityRate, TF_DataType.TF_DOUBLE))
                        {
                            _tensorMyStage = tf.constant((float)Stage.Mortality) * _tensorMyStage / (float)Stage.Infected;
                            return (int)Stage.Mortality;
                        }
                        _tensorMyStage = tf.constant((float)Stage.Recovered) * _tensorMyStage / (float)Stage.Infected;
                        return (int)Stage.Recovered;
                    }
                    return (int)Stage.Infected;
                case (int)Stage.Recovered:
                    return (int)Stage.Recovered;
            }
            return (int)Stage.Mortality;
        }
        
        public static void SetLamdaGammaIntegrals()
        {
            var scale = 5.15;
            var rate = 2.14;
            double b = rate * rate / scale;
            double a = scale / b;
            var res = new List<float>();

            for (int t = 1; t <= Params.Steps + 10; t++)
            {
                double cdfAtTimeT = Gamma.CDF(a, b, t);
                double cdfAtTimeTMinusOne = Gamma.CDF(a, b, t - 1);
                res.Add((float)(cdfAtTimeT - cdfAtTimeTMinusOne));
            }
            _lamdaGammaIntegrals = res.ToArray();
        }

        private InfectionLayer _infectionLayer;

        public Guid ID { get; set; }
    }
    
    
}
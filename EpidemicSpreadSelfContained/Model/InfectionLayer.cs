using System;
using System.Linq;
using Mars.Components.Layers;
using Mars.Core.Data;
using Mars.Interfaces.Data;
using Mars.Interfaces.Layers;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadSelfContained.Model

{
    public class InfectionLayer : AbstractLayer, ISteppedActiveLayer
    {
        public ContactGraphEnvironment ContactEnvironment { get; private set; }
        
        public Tensor Stages { get; set; }

        public Tensor Deaths { get; private set; }

        public IAgentManager AgentManager { get; private set; }

        private LearnableParams _learnableParams;

        private Tensor _infectedIndex;

        private Tensor _infectedTime;

        private Tensor[] _nextStageTimes;
        
        private Tensor[] _stages;

        // private Tensor[] _hostsRecoveredOrDead;


        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            Host.SetLamdaGammaIntegrals();
            ContactEnvironment = new ContactGraphEnvironment();
            // _hostsRecoveredOrDead = new Tensor[Params.AgentCount];
            _stages = new Tensor[Params.AgentCount];
            _nextStageTimes = new Tensor[Params.AgentCount];
            AgentManager = layerInitData.Container.Resolve<IAgentManager>();
            AgentManager.Spawn<Host, InfectionLayer>().ToList();
            ContactEnvironment.ReadCSV();
            _learnableParams = LearnableParams.Instance;
            Deaths = tf.constant(0f);
            Stages = tf.ones(new Shape(Params.AgentCount, 1));
            return initiated;
        }

        public void Tick()
        {
        }

        public void PreTick()
        {
        }
        
        public void PostTick()
        {
            
            // for (int i = 0; i < Params.AgentCount; i++)
            // {
            //     recoveredAndDead += _hosts[i].RecoveredOrDead;
            // }

            var stages = tf.stack(_stages);
            var recoveredAndDead= stages * tf.equal(tf.cast(stages, TF_DataType.TF_INT32), 
                tf.constant(Stage.Infected, TF_DataType.TF_INT32)) * tf.less_equal(_nextStageTimes, 
                (int)Context.CurrentTick) / (float)Stage.Infected;
            
            Deaths += tf.reduce_sum(recoveredAndDead) * _learnableParams.MortalityRate;
            
            // var recoveredAndDead = tf.stack(_hostsRecoveredOrDead); 
            // Deaths += recoveredAndDead * _learnableParams.MortalityRate;
        }

        public void Insert(int index, Tensor stage, Tensor nextStageTime)
        {
            _stages[index] = stage;
            _nextStageTimes[index] = nextStageTime;
        }
        
        // public void Insert(int index, Tensor recoveredOrDead)
        // {
        //     _hostsRecoveredOrDead[index] = recoveredOrDead;
        // }
    }
}
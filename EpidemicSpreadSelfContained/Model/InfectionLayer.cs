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

        private Tensor _nextStageTimes;

        private Host[] _hosts;


        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            Host.SetLamdaGammaIntegrals();
            ContactEnvironment = new ContactGraphEnvironment();
            _hosts = new Host[Params.AgentCount];
            AgentManager = layerInitData.Container.Resolve<IAgentManager>();
            AgentManager.Spawn<Host, InfectionLayer>().ToList();
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
            var recoveredAndDead = tf.constant(0f);
            
            for (int i = 0; i < Params.AgentCount; i++)
            {
                recoveredAndDead += _hosts[i].RecoveredOrDead;
            }
            
            Deaths += recoveredAndDead * _learnableParams.MortalityRate;
        }

        public void Insert(Host host)
        {
            _hosts[host.Index] = host;
        }
    }
}
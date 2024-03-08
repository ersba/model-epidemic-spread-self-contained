using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadSelfContained
{
    public static class Params
    {
        public static readonly int ChildUpperIndex = 1;
        
        public static readonly int AdultUpperIndex = 6;
        
        public static readonly int[] Mu = { 2, 4, 3 };

        public static int Steps = 0;

        public static int AgentCount = 0;

        public static string ContactEdgesPath = "Resources/contact_edges_five_to_fifteen.csv";

        public static string OptimizedParametersPath = "Resources/optimized_parameters.csv";
        
        public static Tensor EdgeAttribute = tf.constant(1f);
        
        public static Tensor Susceptibility = tf.constant(new [] {0.35f, 0.69f, 1.03f, 1.03f, 1.03f, 1.03f, 1.27f, 1.52f});
        
        public static Tensor Infector = tf.constant(new [] {0.0f, 0.33f, 0.72f, 0.0f, 0.0f});

        public static Tensor R0Value = tf.constant(5.18, dtype: TF_DataType.TF_FLOAT);
        
        public static Tensor ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_INT32);
            
        public static Tensor InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_INT32);
    } 
}
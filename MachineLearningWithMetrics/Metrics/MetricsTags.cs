using App.Metrics;

namespace MachineLearningWithMetrics.Metrics
{
    public static class MetricsTags
    {
        private static string[] keys = new string[]
        {
            "Network",
            "Algorithm",
            "MetricType"
        };

        public static MetricTags CreateMetricsTags(string[] keys, string[] values)
        {
            return new MetricTags(keys, values);
        }

        public static MetricTags CreateMetricsTags(string[] values)
        {
            return new MetricTags(keys, values);
        }
    }
}

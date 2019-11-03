using App.Metrics;

namespace MachineLearningWithMetrics.Metrics
{
    public static class MetricsTags
    {
        public static MetricTags CreateMetricsTags(string[] keys, string[] values)
        {
            return new MetricTags(keys, values);
        }
    }
}

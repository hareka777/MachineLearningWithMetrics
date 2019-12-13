using App.Metrics;
using App.Metrics.Gauge;
using App.Metrics.ReservoirSampling.ExponentialDecay;
using App.Metrics.Timer;

namespace MachineLearningWithMetrics.Metrics
{
    /*
     * Static class to handle different metrics objects
     * These objects can be inkoved to be able to log metrics     * 
     */
    public static class MetricsRegistry
    {

        public static TimerOptions Timer => new TimerOptions
        {
            Name = "Timer",
            MeasurementUnit = Unit.Items,
            DurationUnit = TimeUnit.Milliseconds,
            RateUnit = TimeUnit.Milliseconds,
        };

        public static GaugeOptions MemoryUsage => new GaugeOptions
        {
            Name = "Current Memory Usage",
            MeasurementUnit = Unit.MegaBytes
        };

        public static GaugeOptions CPUUsage => new GaugeOptions
        {
            Name = "Current CPU Usage",
            MeasurementUnit = Unit.Percent
        };

        public static GaugeOptions NetworkEvaluatingResult => new GaugeOptions
        {
            Name = "Network Evaluating Result",
            MeasurementUnit = Unit.None
        };

        public static GaugeOptions TrainTestRate => new GaugeOptions
        {
            Name = "TrainTestRate",
            MeasurementUnit = Unit.None
        };

    }
}

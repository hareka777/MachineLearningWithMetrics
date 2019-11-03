using App.Metrics;
using App.Metrics.Gauge;
using App.Metrics.ReservoirSampling.ExponentialDecay;
using App.Metrics.Timer;

namespace MachineLearningWithMetrics.Metrics
{
    public static class MetricsRegistry
    {
        public static TimerOptions Timer => new TimerOptions
        {
            Name = "Timer",
            MeasurementUnit = Unit.Items,
            DurationUnit = TimeUnit.Milliseconds,
            RateUnit = TimeUnit.Milliseconds,
            //Reservoir = () => new DefaultForwardDecayingReservoir(sampleSize: 1028, alpha: 0.015)
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

    }
}

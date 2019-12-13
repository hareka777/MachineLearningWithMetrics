using App.Metrics;
using App.Metrics.Scheduling;
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;



namespace MachineLearningWithMetrics.Metrics
{
    /*
     * Static class for the metrics object
     */
    public static class MetricsInitializer 
    {
        #region Fields

        //Metrics object
        public static IMetricsRoot Metrics;

        //Thread for machine metrics
        private static Thread machineMetricsThread = new Thread(new ThreadStart(LogMachineMetrics));

        #endregion

        #region Constructor
        static MetricsInitializer()
        {
            InitializeMetrics();
        }
        #endregion
        /*
         * Initializing metrics
         */

        #region Methods
        private static void InitializeMetrics()
        {
            //Building the metrics object and setting host and port
            Metrics = AppMetrics.CreateDefaultBuilder()
                .Report.ToInfluxDb(options =>
                {
                    options.InfluxDb.BaseUri = new Uri("http://127.0.0.1:8086/");
                    options.InfluxDb.Database = "sample-db";
                    options.InfluxDb.CreateDataBaseIfNotExists = true;
                })
                .Configuration.Configure(options =>
                {
                    options.AddServerTag();
                })
               .Build();
            
            //Start logging metrics
            var scheduler = new AppMetricsTaskScheduler(
            TimeSpan.FromSeconds(5),
             async () =>
             {
                 await Task.WhenAll(Metrics.ReportRunner.RunAllAsync());
             });
            scheduler.Start();

            machineMetricsThread.Start();
        }

        /*
         * Thread function for logging metrics
         */
        private static void LogMachineMetrics()
        {
            while (true)
            {
                double totalCpu = Process.GetCurrentProcess().TotalProcessorTime.TotalMilliseconds;
                double usedCpu = Process.GetCurrentProcess().PrivilegedProcessorTime.TotalMilliseconds;
                double noOfCpus = Environment.ProcessorCount;
                double cpuUsage = usedCpu / totalCpu / noOfCpus * 100;
                Metrics.Measure.Gauge.SetValue(MetricsRegistry.CPUUsage, cpuUsage);
                Metrics.Measure.Gauge.SetValue(MetricsRegistry.MemoryUsage, Process.GetCurrentProcess().WorkingSet64 / 1024.0 / 1024.0);
                Thread.Sleep(3000);
            }
            
        }
        #endregion
    }
}

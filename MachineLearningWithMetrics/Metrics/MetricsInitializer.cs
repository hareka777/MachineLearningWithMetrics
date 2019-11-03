using App.Metrics;
using App.Metrics.Scheduling;
using System;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;



namespace MachineLearningWithMetrics.Metrics
{
    public static class MetricsInitializer 
    {
        public static IMetricsRoot Metrics;

        private static Thread machineMetricsThread = new Thread(new ThreadStart(LogMachineMetrics));
        
        static MetricsInitializer()
        {
            InitializeMetrics();
        }
        private static void InitializeMetrics()
        {

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
                    //giving the current class name
                    options.GlobalTags.Add("Class", MethodBase.GetCurrentMethod().DeclaringType.Name);
                })
               .Build();
            
            
            var scheduler = new AppMetricsTaskScheduler(
            TimeSpan.FromSeconds(3),
             async () =>
             {
                 await Task.WhenAll(Metrics.ReportRunner.RunAllAsync());
             });
            scheduler.Start();

            machineMetricsThread.Start();
        }

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
    }
}

using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.MNIST;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public class MNISTPredictor
    {
        #region Fields
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\MNIST";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath;
        private readonly string networkPath = Paths.networkModelFolderPath + @"\MNIST.zip";

        private string trainingDataPath = dataFolderPath + @"\MNISTTrainingShort.csv";
        private string testDataPath = dataFolderPath + @"\MNISTTest.csv";

        IMetricsRoot _metrics;
        #endregion

        #region Constructor

        public MNISTPredictor()
        {
            this._metrics = MetricsInitializer.Metrics;
            ProcessNetwork();
        }

        #endregion

        #region Private Methods
        private void ProcessNetwork()
        {
            IDataView trainingData = null;
            IDataView testData = null;
            EstimatorChain<KeyToValueMappingTransformer> pipeline = null;
            ITransformer trainedModel = null;
            MLContext mlContext = new MLContext();
            try{
                Dictionary<string, string> tags = new Dictionary<string, string>();
                tags.Add("ProcessName", "LoadingData");
                tags.Add("TaskName", "MNIST");

                var metricsTags = new MetricTags(tags.Keys.ToArray(), tags.Values.ToArray());

                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, metricsTags))
                {
                    trainingData = LoadData(mlContext, trainingDataPath);
                    testData = LoadData(mlContext, testDataPath);
                }

                
                //MessageBox.Show("Data successfully loaded!");
            }
            catch(Exception e)
            {
                MessageBox.Show("Could not load data: " + e.Message);
            }
            try
            {
                //pipeline = ConfigureNetwork(mlContext);
            }
            catch(Exception e)
            {
                MessageBox.Show("Could not set pipeline: " + e.Message);
            }
            try
            {
                Dictionary<string, string> tags = new Dictionary<string, string>();
                tags.Add("ProcessName", "Training");
                tags.Add("TaskName", "MNIST");

                MetricTags metricsTags = new MetricTags(tags.Keys.ToArray(), tags.Values.ToArray());
                LogMemoryMetrics(metricsTags);
                LogCPUUsage(metricsTags);
                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, metricsTags))
                {
                    //trainedModel = Train(pipeline, trainingData);
                }
                }
            catch(Exception e)
            {
                MessageBox.Show("Training was not successful: " + e.Message);
            }
            try
            {
                //EvaluateModel(trainedModel, mlContext, testData);
            }
            catch (Exception e)
            {
                MessageBox.Show("Evaluating the model was not successful: " + e.Message);
            }
            try
            {
                //SaveNetwork(mlContext, trainedModel, trainingData);
            }
            catch(Exception e)
            {
                MessageBox.Show("Saving the network was not successful: " + e.Message);
            }
            try
            {
                TestSomePredictions(mlContext);
            }
            catch(Exception e)
            {
                MessageBox.Show("Loading and using the network was not successful: " + e.Message);
            }
            
        }

        public IDataView LoadData(MLContext context, string dataPath)
        {
            IDataView loadedData = null;
            //loading TrainingData
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                loadedData = context.Data.LoadFromTextFile(path: dataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
                            //DataKind: size of data in bytes
                            new TextLoader.Column("Number", DataKind.Single, 64)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );
            }
                
            return loadedData;
            
        }

        public EstimatorChain<KeyToValueMappingTransformer> ConfigureNetwork(MLContext context)
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.
            var dataProcessPipeline = context.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.Concatenate("Features", nameof(InputData.PixelValues))
                .AppendCacheCheckpoint(context));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Number", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(context.Transforms.Conversion.MapKeyToValue("Number", "Label"));

            return trainingPipeline;
        }

        public ITransformer Train(EstimatorChain<KeyToValueMappingTransformer> pipeline, IDataView data)
        {
            return pipeline.Fit(data);
        }

        public void EvaluateModel(ITransformer trainedModel, MLContext context, IDataView testData)
        {
            var predictions = trainedModel.Transform(testData);
            var metrics = context.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
            Console.WriteLine("Confusion matrix: " + metrics.ConfusionMatrix.ToString());
        }

        public void SaveNetwork(MLContext context, ITransformer trainedModel, IDataView trainingData)
        {
            context.Model.Save(trainedModel, trainingData.Schema, networkPath);
        }

        private void TestSomePredictions(MLContext mlContext)
        {
            //out var: 
            ITransformer trainedModel;

            Dictionary<string, string> tags = new Dictionary<string, string>();
            tags.Add("ProcessName", "LoadingNetwork");
            tags.Add("TaskName", "MNIST");

            var metricsTags = new MetricTags(tags.Keys.ToArray(), tags.Values.ToArray());

            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, metricsTags))
            {
                trainedModel = mlContext.Model.Load(networkPath, out var modelInputSchema);
            }

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(trainedModel);

            var resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);
            ShowPrediction(1,resultprediction1);           
            
            var resultprediction2 = predEngine.Predict(SampleMNISTData.MNIST2);
            ShowPrediction(7,resultprediction2);

            var resultprediction3 = predEngine.Predict(SampleMNISTData.MNIST3);
            ShowPrediction(9,resultprediction3);
          
        }

        private void ShowPrediction(int actual,OutputData result)
        {
            MessageBox.Show($"Actual: {actual} \n" +
                    $"Predicted probability:\n" +
                    $"zero:  {result.Score[0]:0.####} \n" +
                    $"One :  {result.Score[1]:0.####} \n" +
                    $"two:   {result.Score[2]:0.####} \n" +
                    $"three: {result.Score[3]:0.####} \n" +
                    $"four:  {result.Score[4]:0.####} \n" +
                    $"five:  {result.Score[5]:0.####} \n" +
                    $"six:   {result.Score[6]:0.####} \n" +
                    $"seven: {result.Score[7]:0.####} \n" +
                    $"eight: {result.Score[8]:0.####} \n" +
                    $"nine:  {result.Score[9]:0.####}");
        }

        private void LogMemoryMetrics(MetricTags tags)
        {
            Process process = Process.GetCurrentProcess();
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.MemoryUsage, process.WorkingSet64/ 1024.0 / 1024.0);
        }

        private void LogCPUUsage(MetricTags tags)
        {
            double totalCpu = Process.GetCurrentProcess().TotalProcessorTime.TotalMilliseconds;
            double usedCpu = Process.GetCurrentProcess().PrivilegedProcessorTime.TotalMilliseconds;
            double noOfCpus = Environment.ProcessorCount;
            double cpuUsage = usedCpu / totalCpu / noOfCpus*100;            
            
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.CPUUsage, cpuUsage);
        }
        #endregion
    }
}

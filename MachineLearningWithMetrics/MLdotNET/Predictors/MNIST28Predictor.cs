using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.MNIST28;
using MachineLearningWithMetrics.ViewModels;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using static MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes.Algorithms;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public class MNIST28Predictor : IPredictor
    {
        #region Fields
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\MNIST";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath;
        private readonly string networkPath = Paths.networkModelFolderPath + @"\MNIST28.zip";

        private string trainingDataPath = dataFolderPath + @"\MnistTrainingShort28.csv";
        private string testDataPath = dataFolderPath + @"\Mnist28TestData.csv";

        MultiClassificationTrainingAlgo trainingAlgo;

        IMetricsRoot _metrics;
        #endregion

        #region Constructors
        public MNIST28Predictor()
        {
            this._metrics = MetricsInitializer.Metrics;
            trainingAlgo = MultiClassificationTrainingAlgo.NaiveBayes;
            ProcessNetwork();
        }

        public MNIST28Predictor(MultiClassificationTrainingAlgo algo)
        {
            this._metrics = MetricsInitializer.Metrics;
            trainingAlgo = algo;
            ProcessNetwork();
        }
        #endregion

        #region Implementing Abstract Methods

        internal override void ProcessNetwork()
        {
            EstimatorChain<KeyToValueMappingTransformer> pipeline = null;

            try
            {
                Dictionary<string, string> tags = new Dictionary<string, string>();
                tags.Add("ProcessName", "LoadingData");
                tags.Add("TaskName", "MNIST28");

                var metricsTags = new MetricTags(tags.Keys.ToArray(), tags.Values.ToArray());

                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, metricsTags))
                {
                    trainingData = LoadData(mlContext, trainingDataPath);
                    testData = LoadData(mlContext, testDataPath);
                }

                Console.WriteLine(trainingData.Preview().RowView.ToString());

                //MessageBox.Show("Data successfully loaded!");
            }
            catch (Exception e)
            {
                MessageBox.Show("Could not load data: " + e.Message);
            }
            try
            {
                pipeline = ConfigureNetwork();
            }
            catch (Exception e)
            {
                MessageBox.Show("Could not set pipeline: " + e.Message);
            }
            try
            {               
                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
                {
                    trainedModel = Train(pipeline);
                }
            }
            catch (Exception e)
            {
                MessageBox.Show("Training was not successful: " + e.Message);
            }
            try
            {
                EvaluateModel(trainedModel);
            }
            catch (Exception e)
            {
                MessageBox.Show("Evaluating the model was not successful: " + e.Message);
            }
            try
            {
                SaveNetwork(trainedModel);
            }
            catch (Exception e)
            {
                MessageBox.Show("Saving the network was not successful: " + e.Message);
            }
            try
            {
                TestSomePredictions();
            }
            catch (Exception e)
            {
                MessageBox.Show("Loading and using the network was not successful: " + e.Message);
            }

        }
        internal override IDataView LoadData(MLContext context, string dataPath)
        {
            IDataView loadedData = null;
            //loading TrainingData
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                loadedData = context.Data.LoadFromTextFile(path: dataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(MNIST28DataModel.Pixels), DataKind.Single, 0, 783),
                            //DataKind: size of data in bytes
                            new TextLoader.Column("Digit", DataKind.Single, 784)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );
            }
            
            return loadedData;

        }        

        internal override void EvaluateModel(ITransformer trainedModel)
        {
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Digit", scoreColumnName: "Score");
            Console.WriteLine("Confusion matrix: " + metrics.ConfusionMatrix.ToString());
        }

        internal override void SaveNetwork(ITransformer trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainingData.Schema, networkPath);
        }

        internal override void TestSomePredictions()
        {
            //out var: 
            ITransformer trainedModel;

            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                trainedModel = mlContext.Model.Load(networkPath, out var modelInputSchema);
            }

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<MNIST28DataModel, MNIST28DataModelOutput>(trainedModel);

            var resultprediction0 = predEngine.Predict(Mnist28SampleData.Zero);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel = new DigitDisplayWindowViewModel(Mnist28SampleData.Zero.Pixels);
            digitDisplayWindowViewModel.Show();
            ShowPrediction(0, resultprediction0);

            var resultprediction1 = predEngine.Predict(Mnist28SampleData.One);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel1 = new DigitDisplayWindowViewModel(Mnist28SampleData.One.Pixels);
            digitDisplayWindowViewModel1.Show();
            ShowPrediction(1, resultprediction1);

            var resultprediction2 = predEngine.Predict(Mnist28SampleData.Two);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel2 = new DigitDisplayWindowViewModel(Mnist28SampleData.Two.Pixels);
            digitDisplayWindowViewModel2.Show();
            ShowPrediction(2, resultprediction2);

            var resultprediction3 = predEngine.Predict(Mnist28SampleData.Three);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel3 = new DigitDisplayWindowViewModel(Mnist28SampleData.Three.Pixels);
            digitDisplayWindowViewModel3.Show();
            ShowPrediction(3, resultprediction3);

            var resultprediction4 = predEngine.Predict(Mnist28SampleData.Four);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel4 = new DigitDisplayWindowViewModel(Mnist28SampleData.Four.Pixels);
            digitDisplayWindowViewModel4.Show();
            ShowPrediction(4, resultprediction4);

            var resultprediction5 = predEngine.Predict(Mnist28SampleData.Five);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel5 = new DigitDisplayWindowViewModel(Mnist28SampleData.Five.Pixels);
            digitDisplayWindowViewModel5.Show();
            ShowPrediction(5, resultprediction5);

            var resultprediction6 = predEngine.Predict(Mnist28SampleData.Six);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel6 = new DigitDisplayWindowViewModel(Mnist28SampleData.Six.Pixels);
            digitDisplayWindowViewModel6.Show();
            ShowPrediction(6, resultprediction6);

            var resultprediction7 = predEngine.Predict(Mnist28SampleData.Seven);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel7 = new DigitDisplayWindowViewModel(Mnist28SampleData.Seven.Pixels);
            digitDisplayWindowViewModel7.Show();
            ShowPrediction(7, resultprediction7);

            var resultprediction8 = predEngine.Predict(Mnist28SampleData.Eight);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel8 = new DigitDisplayWindowViewModel(Mnist28SampleData.Eight.Pixels);
            digitDisplayWindowViewModel8.Show();
            ShowPrediction(8, resultprediction8);

            var resultprediction9 = predEngine.Predict(Mnist28SampleData.Nine);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel9 = new DigitDisplayWindowViewModel(Mnist28SampleData.Nine.Pixels);
            digitDisplayWindowViewModel9.Show();
            ShowPrediction(9, resultprediction9);

        }

        private void ShowPrediction(int actual, MNIST28DataModelOutput result)
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
                    $"nine:  {result.Score[9]:0.####} \n" +
                    $"Predicted Value: {CountPredictedDigit(result.Score)}"
                    );
        }

        private int CountPredictedDigit(float[] probalilities)
        {
            float maxProbability = probalilities.FirstOrDefault();
            int predictedDigit = 0;
            for (int i = 1; i < probalilities.Length; i++)
            {
                if (probalilities[i] > maxProbability)
                {
                    maxProbability = probalilities[i];
                    predictedDigit = i;
                }
            }
            return predictedDigit;
        }
        #endregion

        #region Private Methods
        private EstimatorChain<KeyToValueMappingTransformer> ConfigureNetwork()
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            // Use in-memory cache for small/medium datasets to lower training time. Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Digit", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.Concatenate("Features", nameof(MNIST28DataModel.Pixels))
                //Normalizing data
                .Append(mlContext.Transforms.NormalizeLpNorm(outputColumnName: nameof(MNIST28DataModel.Pixels)))
                );

            // STEP 3: Set the training algorithm, then create and config the modelBuilder
            var trainer = ApplyTrainingAlgo(mlContext, trainingAlgo);


            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Digit", "Label"));

            return trainingPipeline;
        }

        private ITransformer Train(EstimatorChain<KeyToValueMappingTransformer> pipeline)
        {
            return pipeline.Fit(trainingData);
        }
        #endregion
    }
}

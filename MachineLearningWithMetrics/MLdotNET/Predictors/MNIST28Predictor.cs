using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.MNIST28;
using MachineLearningWithMetrics.ViewModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Linq;
using System.Windows;
using static MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes.Algorithms;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public class MNIST28Predictor : IPredictor
    {
        #region Fields and Properties
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\MNIST";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath;
        private readonly string networkPath = Paths.networkModelFolderPath + @"\MNIST28.zip";
        private string DataPath = dataFolderPath + @"\MnistTrainingShort1.csv";

        public MultiClassificationTrainingAlgorithm TrainingAlgo
        {
            get;
            set;
        }

        IMetricsRoot _metrics;
        #endregion

        #region Constructors
        public MNIST28Predictor()
        {
            this._metrics = MetricsInitializer.Metrics;
            TrainingAlgo = MultiClassificationTrainingAlgorithm.LbfgsMaximumEntropy;
            this.TrainTestDataRate = 0.2;
        }
        #endregion

        #region Implementing Abstract Methods

        public override void ProcessNetwork()
        {
            EstimatorChain<KeyToValueMappingTransformer> pipeline = null;

            try
            {
                IDataView loadedData = null;
                string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Loading Data"
                };

                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
                {
                    loadedData = LoadData(mlContext, DataPath);
                }

                TrainTestData allData = mlContext.Data.TrainTestSplit(loadedData, testFraction: trainTestDataRate);
                trainingData = allData.TrainSet;
                testData = allData.TestSet;
                //Console.WriteLine(csvData.Preview().RowView.ToString());

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
                string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Training the model"
                };
                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
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
                string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Evaluating Network"
                };
                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
                {
                    EvaluateModel(trainedModel);
                }
            }
                    
            catch (Exception e)
            {
                MessageBox.Show("Evaluating the model was not successful: " + e.Message);
            }
            try
            {
                string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgo.ToString(),
                "Saving Network"
            };
                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
                {
                    SaveNetwork(trainedModel);
                }                    
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


           IDataView loadedData = context.Data.LoadFromTextFile(path: dataPath,
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(MNIST28DataModel.Pixels), DataKind.Single, 0, 783),
                            //DataKind: size of data in bytes
                            new TextLoader.Column("Digit", DataKind.Single, 784)
                        },
                        hasHeader: false,
                        separatorChar: ','
                        );

            return loadedData;

        }

        internal override void EvaluateModel(ITransformer trainedModel)
        {
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Digit", scoreColumnName: "Score");
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.TrainTestRate, MetricsTags.CreateMetricsTags(new string[] { "Network" }, new string[] { nameof(MNIST28Predictor) }), TrainTestDataRate);
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.NetworkEvaluatingResult, MetricsTags.CreateMetricsTags(new string[] {"Network", "Algorithm", "Metric", "TrainTestRate" },new string[] {nameof(MNISTPredictor), this.TrainingAlgo.ToString(), "MacroAccuracy", this.TrainTestDataRate.ToString()}),metrics.MacroAccuracy);
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
            DigitDisplayWindowViewModel digitDisplayWindowViewModel = new DigitDisplayWindowViewModel(Mnist28SampleData.Zero.Pixels, CountPredictedDigit(resultprediction0.Score));
            digitDisplayWindowViewModel.Show();

            var resultprediction1 = predEngine.Predict(Mnist28SampleData.One);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel1 = new DigitDisplayWindowViewModel(Mnist28SampleData.One.Pixels, CountPredictedDigit(resultprediction1.Score));
            digitDisplayWindowViewModel1.Show();

            var resultprediction2 = predEngine.Predict(Mnist28SampleData.Two);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel2 = new DigitDisplayWindowViewModel(Mnist28SampleData.Two.Pixels, CountPredictedDigit(resultprediction2.Score));
            digitDisplayWindowViewModel2.Show();

            var resultprediction3 = predEngine.Predict(Mnist28SampleData.Three);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel3 = new DigitDisplayWindowViewModel(Mnist28SampleData.Three.Pixels, CountPredictedDigit(resultprediction3.Score));
            digitDisplayWindowViewModel3.Show();

            var resultprediction4 = predEngine.Predict(Mnist28SampleData.Four);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel4 = new DigitDisplayWindowViewModel(Mnist28SampleData.Four.Pixels, CountPredictedDigit(resultprediction4.Score));
            digitDisplayWindowViewModel4.Show();

            var resultprediction5 = predEngine.Predict(Mnist28SampleData.Five);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel5 = new DigitDisplayWindowViewModel(Mnist28SampleData.Five.Pixels, CountPredictedDigit(resultprediction5.Score));
            digitDisplayWindowViewModel5.Show();

            var resultprediction6 = predEngine.Predict(Mnist28SampleData.Six);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel6 = new DigitDisplayWindowViewModel(Mnist28SampleData.Six.Pixels, CountPredictedDigit(resultprediction6.Score));
            digitDisplayWindowViewModel6.Show();

            var resultprediction7 = predEngine.Predict(Mnist28SampleData.Seven);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel7 = new DigitDisplayWindowViewModel(Mnist28SampleData.Seven.Pixels, CountPredictedDigit(resultprediction7.Score));
            digitDisplayWindowViewModel7.Show();

            var resultprediction8 = predEngine.Predict(Mnist28SampleData.Eight);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel8 = new DigitDisplayWindowViewModel(Mnist28SampleData.Eight.Pixels, CountPredictedDigit(resultprediction8.Score));
            digitDisplayWindowViewModel8.Show();

            var resultprediction9 = predEngine.Predict(Mnist28SampleData.Nine);
            DigitDisplayWindowViewModel digitDisplayWindowViewModel9 = new DigitDisplayWindowViewModel(Mnist28SampleData.Nine.Pixels, CountPredictedDigit(resultprediction9.Score));
            digitDisplayWindowViewModel9.Show();

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
            var trainer = ApplyMultiTrainingAlgorithm(mlContext, TrainingAlgo);


            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Digit", "Label"));

            return trainingPipeline;
        }

        /*private async Task<ITransformer> Train(EstimatorChain<KeyToValueMappingTransformer> pipeline)
        {
            ITransformer result = null;
            //Thread trainingThread = new Thread(delegate () {
            //    result = TrainThreadFunction(pipeline);
            //});
            //trainingThread.Start();
            //trainingThread.Join();

            result = await Task.Run(() => TrainThreadFunction(pipeline));

            return result;
        }*/
        private ITransformer Train(EstimatorChain<KeyToValueMappingTransformer> pipeline)
        {
            return pipeline.Fit(trainingData);
        }
        #endregion

        public override string ToString()
        {
            return "MNIST(28x28) digit predictor(Multi Classification)";
        }

        public override void SetAlgorithm(object algo)
        {
            this.TrainingAlgo = (MultiClassificationTrainingAlgorithm)algo;
        }

        /*private ITransformer TrainThreadFunction(EstimatorChain<KeyToValueMappingTransformer> pipeline)
        {
            ITransformer trainedModel =  pipeline.Fit(trainingData);
            return trainedModel;
        */
    }
}

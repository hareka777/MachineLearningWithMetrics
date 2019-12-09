using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.BankNotes;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Windows;
using static MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes.Algorithms;

namespace MachineLearningWithMetrics.MLdotNET.Predictors
{
    public class BankNotePredictor : IPredictor
    {
        #region Fields and Properties
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\BankNotes";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath + @"\BankNotes";
        private readonly string networkPath = Paths.networkModelFolderPath + @"\BankNotes.zip";

        private string dataPath = dataFolderPath + @"\data_banknote_authentication.txt";

        IMetricsRoot _metrics;

        public BinaryClassificationTrainingAlgorithm TrainingAlgorithm { get; set; }

        #endregion

        #region Constructors
        public BankNotePredictor()
        {
            _metrics = MetricsInitializer.Metrics;
            this.trainTestDataRate = 0.2;
            TrainingAlgorithm = BinaryClassificationTrainingAlgorithm.LbfgsLogisticRegression;
        }

        #endregion
        #region Implementing abstract methods
        internal override void EvaluateModel(ITransformer trainedModel)
        {
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.TrainTestRate, MetricsTags.CreateMetricsTags(new string[] { "Network" }, new string[] { nameof(BankNotePredictor) }), TrainTestDataRate);
            _metrics.Measure.Gauge.SetValue(MetricsRegistry.NetworkEvaluatingResult, MetricsTags.CreateMetricsTags(new string[] { "Network", "Algorithm", "Metric", "TrainTestRate" }, new string[] { nameof(BankNotePredictor), this.TrainingAlgorithm.ToString(), "Accuracy", this.TrainTestDataRate.ToString() }), metrics.Accuracy);

        }

        internal override IDataView LoadData(MLContext context, string dataPath)
        {
            IDataView loadedData = null;
            //loading TrainingData
            string[] tags = new string[]{
                this.ToString(),
                this.TrainingAlgorithm.ToString(),
                "Loading Data"
            };
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, MetricsTags.CreateMetricsTags(tags)))
            {
                loadedData = context.Data.LoadFromTextFile<BankNotesInput>(path: dataPath,
                        hasHeader: false,
                        separatorChar: ','
                        );
            }

            return loadedData;
        }

        public override void ProcessNetwork()
        {

            EstimatorChain<ITransformer> pipeline = null;

            try
            {                
                IDataView dataView = LoadData(mlContext, dataPath);
                AppendTrainingTestDataRate(dataView);
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
                this.TrainingAlgorithm.ToString(),
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
                this.TrainingAlgorithm.ToString(),
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
                this.TrainingAlgorithm.ToString(),
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

        internal override void SaveNetwork(ITransformer trainedModel)
        {
            mlContext.Model.Save(trainedModel, trainingData.Schema, networkPath);
        }

        internal override void TestSomePredictions()
        {
            ITransformer trainedModel;
            //loading the Network
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                trainedModel = mlContext.Model.Load(networkPath, out var modelInputSchema);
            }
            //testing some predictions
            var predEngine = mlContext.Model.CreatePredictionEngine<BankNotesInput, BankNotesOutput>(trainedModel);

            var resultprediction0 = predEngine.Predict(SampleBankNotesData.Authentic1);
            ShowPrediction("Authentic", resultprediction0);

            var resultprediction1 = predEngine.Predict(SampleBankNotesData.Authentic2);
            ShowPrediction("Authentic", resultprediction1);

            var resultprediction2 = predEngine.Predict(SampleBankNotesData.InAuthentic1);
            ShowPrediction("InAuthentic", resultprediction2);

            var resultprediction3 = predEngine.Predict(SampleBankNotesData.InAuthentic2);
            ShowPrediction("InAuthentic", resultprediction3);
        }
        #endregion

        #region Private Methods
        private EstimatorChain<ITransformer> ConfigureNetwork()
        {
            var preparedData = mlContext.Transforms.Concatenate("Features", "Variance", "Skewness", "Kurtosis", "Entropy")
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Variance)))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Skewness)))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Kurtosis)))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Entropy)));

            //Apply tranier algo
            var trainer = ApplyBinaryTrainingAlgorithm(mlContext, TrainingAlgorithm);

            var pipeline = preparedData.Append(trainer);

            return pipeline;
        }

        private ITransformer Train(EstimatorChain< ITransformer> pipeline)
        {
            return pipeline.Fit(trainingData);
        }
        
        private void ShowPrediction(string actualValue, BankNotesOutput prediction)
        {
            MessageBox.Show($"The actual value is: {actualValue}\n" +
                $"The predicted value is: {prediction.Prediction}");
        }
        #endregion
        public override string ToString()
        {
            return "Bank note validity(Binary Classification)";
        }

        public override void SetAlgorithm(object algo)
        {
            this.TrainingAlgorithm = (BinaryClassificationTrainingAlgorithm)algo;
        }
    }
}

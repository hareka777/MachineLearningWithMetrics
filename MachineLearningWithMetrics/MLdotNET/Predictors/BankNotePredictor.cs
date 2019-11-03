using App.Metrics;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel.BankNotes;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using static Microsoft.ML.DataOperationsCatalog;

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

        private double trainTestDataRate;

        public double TrainTestDataRate
        {
            get { return trainTestDataRate; }
            set { trainTestDataRate = value; }
        }


        #endregion

        #region Constructors
        public BankNotePredictor()
        {
            _metrics = MetricsInitializer.Metrics;
            this.trainTestDataRate = 0.2;
            ProcessNetwork();
        }
        #endregion
        #region Implementing abstract methods
        internal override void EvaluateModel(ITransformer trainedModel)
        {
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("Confusion matrix: " + metrics.ConfusionMatrix);
        }

        internal override IDataView LoadData(MLContext context, string dataPath)
        {
            IDataView loadedData = null;
            //loading TrainingData
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                loadedData = context.Data.LoadFromTextFile<BankNotesInput>(path: dataPath,
                        hasHeader: false,
                        separatorChar: ','
                        );
            }
            //var shuffledData = ShuffleData(loadedData);
            return loadedData;
        }

        internal override void ProcessNetwork()
        {
            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> pipeline = null;

            try
            {
                Dictionary<string, string> tags = new Dictionary<string, string>();
                tags.Add("ProcessName", "LoadingData");
                tags.Add("TaskName", "MNIST28");

                var metricsTags = new MetricTags(tags.Keys.ToArray(), tags.Values.ToArray());

                using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer, metricsTags))
                {
                    IDataView dataView = LoadData(mlContext, dataPath);
                    AppendTrainingTestDataRate(dataView);
                }

                //Console.WriteLine(trainingData.Preview().RowView.ToString());

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

            var resultprediction0 = predEngine.Predict(SampleBankNotesDatacs.Authentic1);
            ShowPrediction("Authentic", resultprediction0);

            var resultprediction1 = predEngine.Predict(SampleBankNotesDatacs.Authentic2);
            ShowPrediction("Authentic", resultprediction1);

            var resultprediction2 = predEngine.Predict(SampleBankNotesDatacs.InAuthentic1);
            ShowPrediction("InAuthentic", resultprediction2);

            var resultprediction3 = predEngine.Predict(SampleBankNotesDatacs.InAuthentic2);
            ShowPrediction("InAuthentic", resultprediction3);
        }
        #endregion

        #region Private Methods
        private EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> ConfigureNetwork()
        {
            var preparedData = mlContext.Transforms.Concatenate("Features", "Variance", "Skewness", "Kurtosis", "Entropy")
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Variance)))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Skewness)))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Kurtosis)))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: nameof(BankNotesInput.Entropy)));

            var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features");

            var pipeline = preparedData.Append(trainer);

            return pipeline;
        }

        private ITransformer Train(EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> pipeline)
        {
            return pipeline.Fit(trainingData);
        }

        private void AppendTrainingTestDataRate(IDataView data)
        {
            TrainTestData allData = mlContext.Data.TrainTestSplit(data, testFraction: trainTestDataRate);
            trainingData = allData.TrainSet;
            testData = allData.TestSet;
        }
        
        private void ShowPrediction(string actualValue, BankNotesOutput prediction)
        {
            MessageBox.Show($"The actual value is: {actualValue}\n" +
                $"The predicted value is: {prediction.Prediction}");
        }
        #endregion
    }
}

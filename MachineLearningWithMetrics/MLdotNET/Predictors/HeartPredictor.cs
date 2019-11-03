using App.Metrics;
using MachineLearningWithMetrics.Data;
using MachineLearningWithMetrics.Metrics;
using MachineLearningWithMetrics.MLdotNET.DataModel;
using Microsoft.ML;
using System;
using System.Windows;

namespace MachineLearningWithMetrics.MLdotNET
{
    public class HeartPredictor
    {
        #region Fields
        private static readonly string dataFolderPath = Paths.dataFolderPath + @"\HeartDisease";
        private readonly string dataModelFolderPath = Paths.dataModelFolderPath + @"\HeartDisease";
        private readonly string networkModelFolderPath = Paths.networkModelFolderPath + @"\HeartClassification.zip";

        private string trainingDataPath = dataFolderPath + @"\HeartDiseaseTraining.csv";
        private string testDataPath = dataFolderPath + @"\HeartDisease.csv";

        IMetricsRoot _metrics;

        #endregion

        #region Constructor
        public HeartPredictor()
        {
            _metrics = MetricsInitializer.Metrics;
        }
        #endregion

        #region Public Methods

        public string GetPath()
        {
            string toReturn = string.Empty;
            using (_metrics.Measure.Timer.Time(MetricsRegistry.Timer))
            {
                LoadData();
            }


            return toReturn;
        }

        #endregion

        #region Private Methods
        private void LoadData()
        {
            var mlContext = new MLContext();

            // STEP 1: Common data loading configuration
            try
            {
                var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(trainingDataPath, hasHeader: false, separatorChar: ';');
                var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(testDataPath, hasHeader: false, separatorChar: ';');




                // STEP 2: Concatenate the features and set the training algorithm

                //var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                // .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
                //var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal");
                //.Append(mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features"));
                //FastTree fastTree = new FastTree();
                //var pipeline = fastTree.SetPipelineAlgo(mlContext, mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal"));
                // STEP 3: Training the model
                //ITransformer trainedModel = pipeline.Fit(trainingDataView);
                /*MessageBox.Show("Data successfully loaded and trained!");

                // STEP 4: Evaluate model

                var predictions = trainedModel.Transform(testDataView);
                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions);

                MessageBox.Show($"Metrics for {trainedModel.ToString()}\n\n" +
                        $" binary classification model:\n" +
                        $"Accuracy: {metrics.Accuracy:P2}\n" +
                        $"Area Under Roc Curve:      {metrics.AreaUnderRocCurve:P2}\n" +
                        $"Area Under PrecisionRecall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}\n" +
                        $"F1Score:  {metrics.F1Score:P2}\n" +
                        $"LogLoss:  {metrics.LogLoss:#.##}\n" +
                        $"LogLossReduction:  {metrics.LogLossReduction:#.##}\n" +
                        $"PositivePrecision:  {metrics.PositivePrecision:#.##}\n" +
                        $"PositiveRecall:  {metrics.PositiveRecall:#.##}\n" +
                        $"NegativePrecision:  {metrics.NegativePrecision:#.##}\n" +
                        $"NegativeRecall:  {metrics.NegativeRecall:P2}");*/
                try
                {
                    //mlContext.Model.Save(trainedModel, trainingDataView.Schema, networkModelFolderPath);
                    MessageBox.Show("Network successfully saved!");
                }
                catch (Exception e)
                {
                    MessageBox.Show($"Could not save the network!\n\n{e.Message}");
                }
            }
            catch (Exception e)
            {
                MessageBox.Show($"Could not train the model.\n{e.Message}");
            }
            // STEP 5: Prediction
            //using (_metrics.Measure.Timer.Time(MetricsRegistry.SampleTimer, "Prediction"))
            {
                //TestPrediction(mlContext);
            }



        }

        private void TestPrediction(MLContext mlContext)
        {
            ITransformer trainedModel = mlContext.Model.Load(networkModelFolderPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(trainedModel);

            foreach (var heartData in HeartSampleData.heartDataList)
            {
                var prediction = predictionEngine.Predict(heartData);

                MessageBox.Show($"Prediction:\n" +
                    $"Age: {heartData.Age}\n" +
                    $"Sex: {heartData.Sex}\n" +
                    $"Cp: {heartData.Cp}\n" +
                    $"TrestBps: {heartData.TrestBps}\n" +
                    $"Chol: {heartData.Chol}\n" +
                    $"Fbs: {heartData.Fbs}\n" +
                    $"RestEcg: {heartData.RestEcg}\n" +
                    $"Thalac: {heartData.Thalac}\n" +
                    $"Exang: {heartData.Exang}\n" +
                    $"OldPeak: {heartData.OldPeak}\n" +
                    $"Slope: {heartData.Slope}\n" +
                    $"Ca: {heartData.Ca}\n" +
                    $"Thal: {heartData.Thal}\n" +
                    $"Prediction Value: {prediction.Prediction}\n" +
                    $"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease")}({prediction.Prediction})\n" +
                    $"Probability: {prediction.Probability}");

            }

        }
        #endregion

    }
}

using Microsoft.ML;
using System.ComponentModel;

namespace MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes
{
    public static class Algorithms
    {
        #region Enums
        public enum MultiClassificationTrainingAlgorithm
        {
            [Description("Sdca Maximum Entropy")]
            SdcaMaximumEntropy,
            [Description("Lbfgs Maximum Entropy")]
            LbfgsMaximumEntropy,
            [Description("Naive Bayes")]
            NaiveBayes,
            [Description("Sdca Non Calibrated")]
            SdcaNonCalibrated
        }

        public enum BinaryClassificationTrainingAlgorithm
        {
            [Description("Fast Tree")]
            FastTree,
            [Description("Lbfgs Logistic Regression")]
            LbfgsLogisticRegression,
            [Description("Field Aware Factorization Machine")]
            FieldAwareFactorizationMachine,
            [Description("Gam")]
            Gam
        }

        public enum RegressionTrainingAlgorithm
        {
            [Description("Fast Tree")]
            FastTree,
            [Description("Fast Tree Tweedie")]
            FastTreeTweedie,
            [Description("Gam")]
            Gam
        }
        #endregion

        #region Public Methods
        public static IEstimator<ITransformer> ApplyMultiTrainingAlgorithm(MLContext mlContext, MultiClassificationTrainingAlgorithm trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {
                case MultiClassificationTrainingAlgorithm.SdcaMaximumEntropy:
                    return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
                case MultiClassificationTrainingAlgorithm.LbfgsMaximumEntropy:
                    return mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
                case MultiClassificationTrainingAlgorithm.NaiveBayes:
                    return mlContext.MulticlassClassification.Trainers.NaiveBayes();
                case MultiClassificationTrainingAlgorithm.SdcaNonCalibrated:
                    return mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated();
                default:
                    return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");

            }
        }

        public static IEstimator<ITransformer> ApplyBinaryTrainingAlgorithm(MLContext mlContext, BinaryClassificationTrainingAlgorithm trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {
                case BinaryClassificationTrainingAlgorithm.FastTree:
                    return mlContext.BinaryClassification.Trainers.FastTree();
                case BinaryClassificationTrainingAlgorithm.LbfgsLogisticRegression:
                    return mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();
                case BinaryClassificationTrainingAlgorithm.FieldAwareFactorizationMachine:
                    return mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine();
                case BinaryClassificationTrainingAlgorithm.Gam:
                    return mlContext.BinaryClassification.Trainers.Gam();
                default:
                    return mlContext.BinaryClassification.Trainers.FastTree();

            }
        }

        public static IEstimator<ITransformer> ApplyRegressionTrainingAlgorithm(MLContext mlContext, RegressionTrainingAlgorithm trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {                
                case RegressionTrainingAlgorithm.FastTree:
                    return mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features");
                case RegressionTrainingAlgorithm.FastTreeTweedie:
                    return mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");
                case RegressionTrainingAlgorithm.Gam:
                    return mlContext.Regression.Trainers.Gam(labelColumnName: "Label", featureColumnName: "Features");
                default:
                    return mlContext.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Label", featureColumnName: "Features");

            }
        }
        #endregion
    }
}

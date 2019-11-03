using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningWithMetrics.MLdotNET.Predictors.Common_Classes
{
    public static class Algorithms
    {
        #region Enums
        public enum MultiClassificationTrainingAlgo
     
        {
            SdcaMaximumEntropy,
            LbfgsMaximumEntropy,
            NaiveBayes,
            SdcaNonCalibrated
        }
        #endregion

        #region Public Methods
        public static IEstimator<ITransformer> ApplyTrainingAlgo(MLContext mlContext, MultiClassificationTrainingAlgo trainingAlgorithm)
        {
            switch (trainingAlgorithm)
            {
                case MultiClassificationTrainingAlgo.SdcaMaximumEntropy:
                    return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
                case MultiClassificationTrainingAlgo.LbfgsMaximumEntropy:
                    return mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
                case MultiClassificationTrainingAlgo.NaiveBayes:
                    return mlContext.MulticlassClassification.Trainers.NaiveBayes();
                case MultiClassificationTrainingAlgo.SdcaNonCalibrated:
                    return mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated();
                default:
                    return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");

            }
        }
        #endregion
    }
}

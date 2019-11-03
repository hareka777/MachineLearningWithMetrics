using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.Eurorate
{
    public class EuroDataModelOutput
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public float Prediction;

        // No need to specify ColumnName attribute, because the field
        // name "Probability" is the column name we want.
        [ColumnName("Probability")]
        public float Probability;
        [ColumnName("Score")]
        public float Score;
    }
}

using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.MNIST28
{
    public class MNIST28DataModelOutput
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}

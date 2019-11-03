using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.Eurorate
{
    public class SampleEuroData
    {
        public static readonly EuroDataModel Euro1 = new EuroDataModel()
        {
            DiffPrev1 = 1.0000f,
            DiffPrev2 = 1.0001f,
            DiffPrev3 = 1.0011f,
            DiffPrev4 = 1.0002f,
            DiffPrev5 = 1.0014f,
            DiffPrev6 = 1.0016f,
            DiffPrev7 = 0.9998f,
            DiffPrev8 = 0.9998f,
            DiffPrev9 = 1.0010f,
            DiffPrev10 = 1.0007f

        };//326.0800

        static readonly EuroDataModel Euro2 = new EuroDataModel()
        {
            DiffPrev1 = 0.9997f,
            DiffPrev2 = 1.0001f,
            DiffPrev3 = 1.0011f,
            DiffPrev4 = 1.0002f,
            DiffPrev5 = 1.0014f,
            DiffPrev6 = 1.0016f,
            DiffPrev7 = 0.9998f,
            DiffPrev8 = 0.9998f,
            DiffPrev9 = 1.0010f,
            DiffPrev10 = 1.0007f

        };//314.4700



    }
}

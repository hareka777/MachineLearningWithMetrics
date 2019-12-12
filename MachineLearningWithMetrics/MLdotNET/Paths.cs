using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningWithMetrics.MLdotNET
{
    /*
     * Static class to provide main paths in the project 
     */
    public static class Paths
    {
        private static string mlFolderPath = Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory())) + @"\MLdotNET";
        public static string dataFolderPath = mlFolderPath + @"\Data";
        public static string dataModelFolderPath = mlFolderPath + @"\DataModel";
        public static string networkModelFolderPath = mlFolderPath + @"\Networks";
    }
}

package WekaCluster;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaCluster {
    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/iris.arff");
            Instances data = source.getDataSet();
            // data.setClass(data.attribute("class"));
            System.out.println(data.numAttributes());

            // SimpleKMeans kmeans = new SimpleKMeans();
            // kmeans.setNumClusters(3);
            // kmeans.buildClusterer(data);

            // ClusterEvaluation ke = new ClusterEvaluation();
            // ke.setClusterer(kmeans);
            // ke.evaluateClusterer(data);

            // // first question, why data.setClassIndex(data.numAttribute()-1)
            // // second question, why create a filter name remove

            // System.out.println(ke.clusterResultsToString());

        } catch(Exception e) {
            System.out.println("Error");
            e.printStackTrace();
        } 
    }
}

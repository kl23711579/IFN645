package WekaClassifiersEffectiveness;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaClassifiersEffectiveness {
    public static void main(String[] args) throws Exception {
        String path = "/Users/d1w1rnna/Documents/work/weka-3-8-4/data/diabetes.arff";
        long startLoadData = System.nanoTime();
        DataSource source = new DataSource(path);
        long endLoadData = System.nanoTime();
        double elapsedLoadData = (endLoadData - startLoadData)/1000000000;
        // elapsedLoadData = elapsedLoadData/1000000000;
        System.out.println("Time for load Data: " + elapsedLoadData);
        Instances data = source.getDataSet();
        data.setClass(data.attribute("class"));
        
        WekaClassifiersEffectiveness w = new WekaClassifiersEffectiveness();
        // w.doClassifier(new NaiveBayes(), data, new Evaluation(data));
        
        // String matlab = "[0.0 1.0; 5.0 0.0]";
        // CostMatrix matrix = CostMatrix.parseMatlab(matlab);
        // w.doClassifierWithCost(new NaiveBayes(), data, new Evaluation(data, matrix));
        w.doAllClassifier(data);       
    }

    public void doAllClassifier(Instances data) throws Exception {
        J48 j48 = new J48();
        NaiveBayes nb = new NaiveBayes();
        Evaluation eval = new Evaluation(data);
        doClassifier(j48, data, eval);
        doClassifier(nb, data, eval);
    }

    public void doClassifier(AbstractClassifier classifier, Instances data, Evaluation eval) throws Exception {
        classifier.buildClassifier(data);

        for(int i = 1; i <= 10; i++ ) {
            Random r = new Random(i);
            eval.crossValidateModel(classifier, data, 10, r);
            System.out.println(classifier.getClass() + " Accuracy = " + eval.correct()/eval.numInstances());
        }
    }

    public void doClassifierWithCost(AbstractClassifier classifier, Instances data, Evaluation eval) throws Exception {
        classifier.buildClassifier(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println(classifier.getClass() + " Avg Cost: " + eval.totalCost());
    }
}
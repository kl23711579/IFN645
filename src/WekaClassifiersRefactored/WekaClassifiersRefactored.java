package WekaClassifiersRefactored;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaClassifiersRefactored {
   public static void main(String[] args) {
    //    WekaClassifiersRefactored w = new WekaClassifiersRefactored();
    //    w.doAllClassification();
        try {
            DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/glass.arff");
            Instances data = source.getDataSet();
            data.setClass(data.attribute("Type"));

            J48 j48 = new J48();
            j48.buildClassifier(data);
            System.out.println(j48.toString());
        } catch(Exception e) {
            System.out.println("Error");
            e.printStackTrace();
        } 
   }
   
   public void doAllClassification() {
       try{
          DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/glass.arff");
          Instances data = source.getDataSet();
          data.setClass(data.attribute("Type"));

          // do classification
          doClassification(new J48(), data, "j48", new String[0]);
          doClassification(new OneR(), data, "oneR", new String[0]);
          doClassification(new IBk(), data, "IBk", new String[0]);
          doClassification(new SMO(), data, "SMO", new String[0]);
          doClassification(new NaiveBayes(), data, "NaiveBayes", new String[0]);
          doClassification(new J48(), data, "j48u", new String[]{"-C", "0.1", "-M", "2"});


       } catch(Exception e){
            System.out.println("Error");
            e.printStackTrace();
       }
   }

   public void doClassification(AbstractClassifier classifier, Instances data, String cMethods, String[] options) throws Exception {
       if(options.length > 0) {
           classifier.setOptions(options);
       }
       classifier.buildClassifier(data);
       Evaluation evaluate = new Evaluation(data);
       evaluate.crossValidateModel(classifier, data, 10, new Random(1));
       System.out.println(cMethods + evaluate.correct()/evaluate.numInstances());
   }
}


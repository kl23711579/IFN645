package FeatureSelection;

import java.util.ArrayList;
import java.util.Random;

import java.io.FileWriter;
import java.io.IOException;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.pmml.jaxbbindings.REALSparseArray;

public class FeatureSelection {
    public static void main(String[] args) throws Exception {
        FeatureSelection wfs = new FeatureSelection();
        DataSource source = new DataSource("data/soil.arff");
        Instances data = source.getDataSet();
        data.setClass(data.attribute("Cover_Type"));
        // wfs.doFilterClassification(data);
        int rank_number = data.numAttributes()-1;

        NaiveBayes nb = new NaiveBayes();
        // CfsSubsetEval cfs = new CfsSubsetEval();
        // BestFirst bf = new BestFirst();
        // wfs.doFilteredClassification(nb, data, cfs, bf);

        // WrapperSubsetEval wrapper = new WrapperSubsetEval();
        // wrapper.setClassifier(new J48());
        // wfs.doFilteredClassification(nb, data, wrapper, bf);

        // create classifier
        ArrayList<AbstractClassifier> aclassifiers = new ArrayList<AbstractClassifier>();
        aclassifiers.add(new J48());
        aclassifiers.add(new NaiveBayes());
        aclassifiers.add(new SMO());
        aclassifiers.add(new OneR());
        aclassifiers.add(new IBk());
        // for(int i = 0; i < aclassifiers.size(); i++) {
        //     String name = aclassifiers.get(i).getClass().getName();
        //     System.out.println(name);
        //     String[] names = name.split("\\.");
        //     System.out.println(names[names.length-1]);
        // }

        // Ranker ranker = new Ranker();
        // ranker.setNumToSelect(7);

        ArrayList<ASEvaluation> asevals = new ArrayList<ASEvaluation>();
        asevals.add(new InfoGainAttributeEval());
        asevals.add(new GainRatioAttributeEval());
        // for(int i = 0; i < asevals.size(); i++) {
        //     String[] name = asevals.get(i).getClass().getName().split("\\.");
        //     System.out.println(name[name.length-1]);
        // }

        Ranker ranker = new Ranker();

        // try{
        //     FileWriter myfile = new FileWriter("test.txt");
        //     for(int i = 0; i < 10; i++){
        //         for(int j =0; j < 10; j++){
        //             String s = String.valueOf(i) + "," + String.valueOf(j) + "\n";
        //             myfile.write(s);
        //         }
        //     }
        //     myfile.close();
        // } catch(IOException e){
        //     e.printStackTrace();
        // }
        
        String result = "";
        try{
            FileWriter myfile = new FileWriter("result.txt");
            myfile.write("Classifier,Evaluation,Correct,ModelTime,Time,NumOfFeature,Features\n");
            for(int i = 0; i < aclassifiers.size(); i++) {
                String[] cnames = aclassifiers.get(i).getClass().getName().split("\\.");
                String cname = cnames[cnames.length-1];
                for(int j = 0; j < asevals.size(); j++) {
                    String[] evalnames = asevals.get(j).getClass().getName().split("\\.");
                    String evalname = evalnames[evalnames.length-1];
                    for(int k = rank_number; k > 0; k--) {
                        System.out.println(cname + "," + evalname + "," + String.valueOf(k));
                        ranker.setNumToSelect(k);
                        long starttime = System.nanoTime();
                        String result2 = wfs.doFilteredClassification(aclassifiers.get(i), data, asevals.get(j), ranker);
                        long endtime = System.nanoTime();
                        double elapsed = (endtime - starttime);
                        elapsed = elapsed/1000000000;
                        result = cname + "," + evalname + "," + result2 + ",";
                        result += String.valueOf(elapsed);
                        result += ",";
                        result += String.valueOf(k);
                        result += ",";
                        result += wfs.doFeatureSelection(data, asevals.get(j), ranker);
                        result += "\n";
                        myfile.write(result);
                    }
                }
            }
            myfile.close();
        } catch(IOException e) {
            System.out.println("Something Errorr");
            e.printStackTrace();
        }

        // for(int i = 0; i < asevals.size(); i++) {
        //     wfs.doFilteredClassification(nb, data, asevals.get(i), ranker);
        // }

        wfs.doFilteredClassification(nb, data, new OneRAttributeEval(), ranker);
        // wfs.doFilteredClassification(nb, data, new InfoGainAttributeEval(), ranker);
        // wfs.doFilteredClassification(nb, data, new GainRatioAttributeEval(), ranker);

    }

    public void doFilterClassification(Instances data) throws Exception{
        // try {
        //     DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/iris.arff");
        //     Instances data = source.getDataSet();
        // } catch(Exception e) {
        //     System.out.println("Error");
        //     e.printStackTrace();
        // }
        doASClassfier(new AttributeSelectedClassifier(), data);
   }

   public String doFilteredClassification(AbstractClassifier classifier, Instances data, ASEvaluation evaluation, ASSearch search) throws Exception{
        
        AttributeSelectedClassifier asc1 = new AttributeSelectedClassifier();
        asc1.setEvaluator(evaluation);
        asc1.setClassifier(classifier);
        asc1.setSearch(search);
        long starttime = System.nanoTime();
        asc1.buildClassifier(data);
        long endtime = System.nanoTime();
        double elapsed = (endtime - starttime);
        elapsed = elapsed/1000000000; 

        Evaluation evaluate = new Evaluation(data);
        evaluate.crossValidateModel(asc1, data, 10, new Random(1));
        String result = String.valueOf(evaluate.correct());
        result = result + "," + String.valueOf(elapsed);
        return result;
   }

    public String doFeatureSelection(Instances data, ASEvaluation evaluation, ASSearch search) throws Exception{
        AttributeSelection as = new AttributeSelection();
        as.setEvaluator(evaluation);
        as.setSearch(search);
        as.SelectAttributes(data);
        int[] indices = as.selectedAttributes();

        String[] s = new String[as.numberAttributesSelected()];
        for(int i = 0; i < indices.length-1; i++) {
            s[i] = data.attribute(indices[i]).name();
        }

        return String.join("-", s);
    }

   public void doASClassfier(AbstractClassifier classifier, Instances data) throws Exception {
        long starttime = System.nanoTime();
        classifier.buildClassifier(data);
        long endtime = System.nanoTime();
        double elapsed = (endtime - starttime);
        elapsed = elapsed/1000000000; 
        Evaluation evaluate = new Evaluation(data);
        evaluate.crossValidateModel(classifier, data, 10, new Random(1));
        System.out.println(evaluate.correct()/evaluate.numInstances());
   }
}
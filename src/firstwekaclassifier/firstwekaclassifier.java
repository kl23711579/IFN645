package firstwekaclassifier;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;;

public class firstwekaclassifier {
    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("/Users/d1w1rnna/Documents/work/weka-3-8-4/data/glass.arff");
            Instances data = source.getDataSet();
            data.setClass(data.attribute("Type"));
            // System.out.println(data);

            J48 j48tree = new J48();
            j48tree.buildClassifier(data);

            // Evaluation
            Evaluation j48eval = new Evaluation(data);
            j48eval.crossValidateModel(j48tree, data, 10, new Random(1));
            System.out.println("J48 Pruned Tree " + j48eval.correct()/j48eval.numInstances());

            J48 j48Utree = new J48();
            String[] j48Uoptions = new String[1];
            j48Uoptions[0] = "-U";
            j48Utree.setOptions(j48Uoptions);
            j48Utree.buildClassifier(data);
            Evaluation j48Ueval = new Evaluation(data);
            j48Ueval.crossValidateModel(j48Utree, data, 10, new Random(1));
            System.out.println("J48 Unpruned Tree " + j48Ueval.correct()/j48Ueval.numInstances());

            // OneR
            OneR oneR = new OneR();
            oneR.buildClassifier(data);
            Evaluation oneReval = new Evaluation(data);
            oneReval.crossValidateModel(oneR, data, 10, new Random(1));
            System.out.println("oneR " + oneReval.correct()/oneReval.numInstances());

            IBk ibk = new IBk();
            ibk.buildClassifier(data);
            Evaluation ibkeval = new Evaluation(data);
            ibkeval.crossValidateModel(ibk, data, 10, new Random(1));
            System.out.println("ibk " + ibkeval.correct()/ibkeval.numInstances());

            SMO smo = new SMO();
            smo.buildClassifier(data);
            Evaluation smoeval = new Evaluation(data);
            smoeval.crossValidateModel(smo, data, 10, new Random(1));
            System.out.println("smo " + smoeval.correct()/smoeval.numInstances());

            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(data);
            Evaluation nbeval = new Evaluation(data);
            nbeval.crossValidateModel(nb, data, 10, new Random(1));
            System.out.println("nb " + nbeval.correct()/nbeval.numInstances());


        } catch (Exception e) {
            System.out.println("Error");
            e.printStackTrace();
        }
    }
}
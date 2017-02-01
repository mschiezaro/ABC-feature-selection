package br.unicamp.ic.execution.comparison;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import br.unicamp.ic.util.FileUtil;

import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.LinearForwardSelection;
import weka.attributeSelection.ScatterSearchV1;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

public class GSWWekaFeatureSelection {
	
	public static void main(String[] args) throws Exception {
		FileUtil fileUtil = FileUtil.newInstance("comparison_GreedyStepWise_feature_selection.log"); 
		GSWWekaFeatureSelection wekaFeatureSelection =  new GSWWekaFeatureSelection();
		ReplaceMissingValues replaceMissingValues =  new ReplaceMissingValues();
		// Filtro z-score
		Standardize zscore = new Standardize();
		GreedyStepwise greedyStepwise = new GreedyStepwise();
		greedyStepwise.setConservativeForwardSelection(false);		

		fileUtil.writeMsg("AutosFeatureSelectionExecution-----------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("BCancerFeatureSelectionExecution---------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", greedyStepwise, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("DiabetesFeatureSelectionExecution--------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("GlassFeatureSelectionExecution-----------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("HeartCFeatureSelectionExecution----------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("HeartStatlogFeatureSelectionExecution----------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("HepaticFeatureSelectionExecution---------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("IrisFeatureSelectionExecution------------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", greedyStepwise, replaceMissingValues));

		fileUtil.writeMsg("LaborFeatureSelectionExecution-----------------------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", greedyStepwise, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("ImageSegmentationFeatureSelectionExecution-----------------------------------------------");
		fileUtil.writeMsg("GreedyStepWise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.close();
	}
	
	public String execute(String databaseName, ASSearch search, Filter... filter) {
		
		Instances instances;
		
		try {
			instances = new Instances(new FileReader(
					System.getProperty("user.dir") + "/src/main/resources/"
							+ databaseName));
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		if (filter != null) {
			for (int i = 0; i < filter.length; i++) {

				try {
					filter[i].setInputFormat(instances);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
				try {
					instances = Filter.useFilter(instances,
							filter[i]);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}

			}
		}
		AttributeSelection selection = new AttributeSelection();
		WrapperSubsetEval eval = new WrapperSubsetEval();
		Classifier knn = new IBk();
		eval.setClassifier(knn);
		eval.setFolds(10);

		selection.setEvaluator(eval);
		selection.setSearch(search);
		selection.setFolds(10);
		selection.setSeed(1);
		try {
			selection.SelectAttributes(instances);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		int[] attributes;
		try {
			attributes = selection.selectedAttributes();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		int featureSize = instances.numAttributes() - 1;
 
		int deletedFetures = 0;
		for (int i = 0; i < featureSize; i++) {
			boolean featureIsIncluded = false;
			for (int j = 0; j < attributes.length; j++) {
				if(i == attributes[j]) {
					featureIsIncluded = true;
					break;
				}
			}
			if (!featureIsIncluded) {
				instances.deleteAttributeAt(i - deletedFetures);
				deletedFetures++;
			}
		}
		instances.setClassIndex(instances.numAttributes() - 1);
		Evaluation evalualtion = null;
		try {
			evalualtion = new Evaluation(instances);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		try {
			evalualtion.crossValidateModel(knn, instances, 10, new Random(1));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		// Retorna o valor da classificação
		return Utils.arrayToString(attributes)+" "+evalualtion.pctCorrect();
	}
}

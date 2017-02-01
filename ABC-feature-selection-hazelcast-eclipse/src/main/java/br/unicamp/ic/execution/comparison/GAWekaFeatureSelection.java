package br.unicamp.ic.execution.comparison;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import br.unicamp.ic.util.FileUtil;

import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.GeneticSearch;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.LinearForwardSelection;
import weka.attributeSelection.PSOSearch;
import weka.attributeSelection.ScatterSearchV1;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

public class GAWekaFeatureSelection {
	
	public static void main(String[] args) throws IOException {
		FileUtil fileUtil = FileUtil.newInstance("comparison_GA_feature_selection.log"); 
		GAWekaFeatureSelection wekaFeatureSelection =  new GAWekaFeatureSelection();
		ReplaceMissingValues replaceMissingValues =  new ReplaceMissingValues();
		// Filtro z-score
		Standardize zscore = new Standardize();
		GeneticSearch geneticSearch = new GeneticSearch();

		fileUtil.writeMsg("AutosFeatureSelectionExecution-----------------------------------------------------------");
		geneticSearch.setCrossoverProb(1);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BCancerFeatureSelectionExecution---------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(70);
		geneticSearch.setMutationProb(0.04);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);

		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", geneticSearch, replaceMissingValues, zscore));

		fileUtil.writeMsg("DiabetesFeatureSelectionExecution--------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("GlassFeatureSelectionExecution-----------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("HeartCFeatureSelectionExecution----------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);

		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", geneticSearch, replaceMissingValues));		

		fileUtil.writeMsg("HeartStatlogFeatureSelectionExecution----------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(40);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", geneticSearch, replaceMissingValues));

		fileUtil.writeMsg("HepaticFeatureSelectionExecution---------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.2);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", geneticSearch, replaceMissingValues));

		fileUtil.writeMsg("IrisFeatureSelectionExecution------------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("LaborFeatureSelectionExecution-----------------------------------------------------------");
		geneticSearch.setCrossoverProb(0.6);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(20);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("ImageSegmentationFeatureSelectionExecution-----------------------------------------------");
		geneticSearch.setCrossoverProb(1);
		geneticSearch.setPopulationSize(20);
		geneticSearch.setMutationProb(0.033);
		geneticSearch.setMaxGenerations(100);
		geneticSearch.setReportFrequency(20);
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", geneticSearch, replaceMissingValues));
		
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

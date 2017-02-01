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

public class WekaFeatureSelection {
	
	public static void main(String[] args) throws IOException {
		FileUtil fileUtil = FileUtil.newInstance("comparison_feature_selection.log"); 
		WekaFeatureSelection wekaFeatureSelection =  new WekaFeatureSelection();
		ReplaceMissingValues replaceMissingValues =  new ReplaceMissingValues();
		// Filtro z-score
		Standardize zscore = new Standardize();
		GeneticSearch geneticSearch = new GeneticSearch();
		PSOSearch psoSearch = new PSOSearch();
		BestFirst bestFirst = new BestFirst();
		GreedyStepwise greedyStepwise = new GreedyStepwise();
		LinearForwardSelection linearForwardSelection = new LinearForwardSelection();
		ScatterSearchV1 scatterSearchV1 = new ScatterSearchV1();
		
		geneticSearch.setPopulationSize(200);
		fileUtil.writeMsg("AutosFeatureSelectionExecution-----------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("autos.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("BCancerFeatureSelectionExecution---------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", geneticSearch, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", psoSearch, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", bestFirst, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", greedyStepwise, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", linearForwardSelection, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("breast-cancer.arff", scatterSearchV1, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("DiabetesFeatureSelectionExecution--------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("diabetes.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("GlassFeatureSelectionExecution-----------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("glass.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("HeartCFeatureSelectionExecution----------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", greedyStepwise, replaceMissingValues));
 
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-c.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("HeartStatlogFeatureSelectionExecution----------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("heart-statlog.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("HepaticFeatureSelectionExecution---------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("hepatitis.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("IrisFeatureSelectionExecution------------------------------------------------------------");
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("iris.arff", scatterSearchV1, replaceMissingValues));
		
		fileUtil.writeMsg("LaborFeatureSelectionExecution-----------------------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", geneticSearch, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg("Z-score");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", psoSearch, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", bestFirst, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", greedyStepwise, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", linearForwardSelection, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("labor.arff", scatterSearchV1, replaceMissingValues, zscore));
		
		fileUtil.writeMsg("ImageSegmentationFeatureSelectionExecution-----------------------------------------------");
		
		fileUtil.writeMsg("GeneticSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", geneticSearch, replaceMissingValues));
		
		fileUtil.writeMsg("PSOSearch----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", psoSearch, replaceMissingValues));
		
		fileUtil.writeMsg("BestFirst----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", bestFirst, replaceMissingValues));
		
		fileUtil.writeMsg("greedyStepwise----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", greedyStepwise, replaceMissingValues));
		
		fileUtil.writeMsg("linearForwardSelection----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", linearForwardSelection, replaceMissingValues));
		
		fileUtil.writeMsg("scatterSearchV1----------------------------------------------------------------------------");
		fileUtil.writeMsg(wekaFeatureSelection.execute("segment.arff", scatterSearchV1, replaceMissingValues));
		
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

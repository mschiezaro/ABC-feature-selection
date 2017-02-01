package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;
import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.AsyncFeatureSelection;
import br.unicamp.ic.featureselection.AsyncFeatureSelection2;
import br.unicamp.ic.util.FileUtil;

public class IrisFeatureSelectionExecution extends FeatureSelectionExecution {

	private AsyncFeatureSelection2 featureSelection;
	
	public IrisFeatureSelectionExecution(boolean[] features) {
		super("iris.arff", features, 20, 6, 0.1, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true};		
		FeatureSelectionExecution fs = new IrisFeatureSelectionExecution(features);
		fs.executeAll();
		FileUtil.newInstance().close();
	}
	
	@Override
	public void executeAll() {
//		executeFullFeaturesWithNoFilters();
		executeWithNoFilter();
//		executeFullFeaturesNormalized();
//		executeWithNormalization();
//		executeFullFeaturesZScore();
//		executeWithZScore();
	}

	@Override
	public void executeWithNormalization() {
		writeMsg("executeWithNormalization");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,
				databaseName, 2, replaceMissingValues, normalize);
		// executa a seleção de atributos
		featureSelection.execute();

	}

	@Override
	public void executeWithZScore() {
		writeMsg("executeWithZScore");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,
				databaseName, 2, replaceMissingValues, zscore);
		// executa a seleção de atributos
		featureSelection.execute();
	}

	@Override
	public void executeWithNoFilter() {
		writeMsg("executeWithNoFilter");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,
				databaseName, 2, replaceMissingValues);
		// executa a seleção de atributos
		featureSelection.execute();
	}
	
	@Override
	public void executeFullFeaturesWithNoFilters() {
		writeMsg("executeFullFeaturesWithNoFilters");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");

	}	
	@Override
	public void executeFullFeaturesNormalized() {
		writeMsg("executeFullFeaturesNormalized");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues, normalize);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");
		
	}
	
	@Override
	public void executeFullFeaturesZScore() {
		writeMsg("executeFullFeaturesZScore");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues, zscore);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");
		
	}
}

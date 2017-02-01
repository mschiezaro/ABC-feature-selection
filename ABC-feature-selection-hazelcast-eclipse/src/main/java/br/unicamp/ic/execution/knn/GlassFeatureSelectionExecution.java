package br.unicamp.ic.execution.knn;

import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.FeatureSelection;
import weka.classifiers.lazy.IBk;

public class GlassFeatureSelectionExecution extends FeatureSelectionExecution {

	private FeatureSelection featureSelection;
	
	public GlassFeatureSelectionExecution(boolean[] features) {
		super("glass.arff", features, 100, 6, 0.1, new IBk());
	}

	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true};
		FeatureSelectionExecution fs = new GlassFeatureSelectionExecution(
				features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();
		executeWithNoFilter();
		executeFullFeaturesNormalized();
		executeWithNormalization();
		executeFullFeaturesZScore();
		executeWithZScore();
	}

	@Override
	public void executeWithNormalization() {
		writeMsg("executeWithNormalization");
		// carrega os atributos e passa os filtros
		featureSelection = new FeatureSelection(runtime, limit, mr,
				databaseName, replaceMissingValues, normalize);
		// executa a seleção de atributos
		featureSelection.execute();

	}

	@Override
	public void executeWithZScore() {
		writeMsg("executeWithZScore");
		// carrega os atributos e passa os filtros
		featureSelection = new FeatureSelection(runtime, limit, mr,
				databaseName, replaceMissingValues, zscore);
		// executa a seleção de atributos
		featureSelection.execute();
	}

	@Override
	public void executeWithNoFilter() {
		writeMsg("executeWithNoFilter");
		// carrega os atributos e passa os filtros
		featureSelection = new FeatureSelection(runtime, limit, mr,
				databaseName, replaceMissingValues);
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

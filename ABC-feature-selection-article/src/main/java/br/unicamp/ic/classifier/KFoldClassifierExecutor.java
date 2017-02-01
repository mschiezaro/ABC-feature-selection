package br.unicamp.ic.classifier;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

public class KFoldClassifierExecutor extends ClassifierExecutor {

	private static final boolean IS_DEBUG_ENABLED = true;

	/**
	 * Classificador
	 */
	private Classifier classifier;

	public KFoldClassifierExecutor(Classifier classifier) {
		this.classifier = classifier;
	}

	/**
	 * Executa a classificação. Os dados de treinamento e teste utilizam k-fold
	 * onde o número de k é indicado pelo parâmetro kfold, e as features que
	 * serão utilizadas na classificação são indicadas pelo vetor
	 * featureInclusion. Cada posição do vetor indica uma feature e o valor true
	 * indica que ela participará da classificação, false não participará
	 * @param featureInclusion  features que serão utilizadas na classificação
	 * @param k Parâmetro k-fold para dividir dados de treinamento e teste
	 */
	public double execute(boolean[] featureInclusion, int kFold) {

		int deletedFetures = 0;
		// Exclui as features que não devem participar do processo de
		// classificação
		for (int i = 0; i < featureInclusion.length; i++) {
			if (!featureInclusion[i]) {
				instances.deleteAttributeAt(i - deletedFetures);
				deletedFetures++;
			}
		}
		instances.setClassIndex(instances.numAttributes() - 1);

		// realiza cross-validation
		Evaluation eval = null;
		try {
			eval = new Evaluation(instances);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		try {
			eval.crossValidateModel(classifier, instances, kFold, new Random(1));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		// Retorna o valor da classificação
		return eval.pctCorrect();
	}

	/**
	 * Executa a classificação. Os dados de treinamento e teste utilizam k-fold
	 * onde o número de k é indicado pelo parâmetro kfold, e as features que
	 * serão utilizadas na classificação são indicadas pelo vetor
	 * featureInclusion. Cada posição do vetor indica uma feature e o valor true
	 * indica que ela participará da classificação, false não participará
	 * @param featureInclusion  features que serão utilizadas na classificação
	 * @param k Parâmetro k-fold para dividir dados de treinamento e teste
	 * @param classIndex indica qual coluna está o label da classificação (ground truth)
	 */
	public double execute(boolean[] featureInclusion, int kFold, int classIndex) {

		int deletedFetures = 0;
		// Exclui as features que não devem participar do processo de
		// classificação
		for (int i = 0; i < featureInclusion.length; i++) {
			if (!featureInclusion[i]) {
				instances.deleteAttributeAt(i - deletedFetures);
				deletedFetures++;
			}
		}
		instances.setClassIndex(classIndex);

		// realiza cross-validation
		Evaluation eval = null;
		try {
			eval = new Evaluation(instances);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		try {
			eval.crossValidateModel(classifier, instances, kFold, new Random(1));
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		// Retorna o valor da classificação
		return eval.pctCorrect();
	}

}

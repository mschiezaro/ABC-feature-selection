package br.unicamp.ic.featureselection;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IExecutorService;
import com.hazelcast.core.Member;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;

import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.swarm.FoodSource;
import br.unicamp.ic.util.FileUtil;

/**
 * Responsável por fazer a seleção de features
 * 
 * @author Mauricio Schiezaro
 */
public class FeatureSelection {

	private BufferedWriter writer;

	/**
	 * Estratégia de execução do classificador
	 */
	private static final int KFOLD = 10;

	/**
	 * Quantidade máximo possível de features
	 */
	private int featureSize;

	/**
	 * Número máximo de vezes que visitaremos uma fonte de alimento antes de
	 * abandoná-la
	 */
	private int limit;

	/**
	 * Número de execuções do algoritmo ABC
	 */
	private int runtime;

	/**
	 * Fitness da melhor fonte de alimento
	 */
	private double bestFitness;

	/**
	 * Melhor fonte de alimento
	 */
	private FoodSource bestFoodSource;

	/**
	 * Parâmetro que controlorá a perturbação de quantas features serão
	 * modificadas ao explorar a vizinhança
	 */
	private double mr;

	/**
	 * Conjunto de fontes de alimento
	 */
	private Set<FoodSource> foodSources;

	/**
	 * Fontes de alimento que já foram visitadas
	 */
	private Set<FoodSource> visitedFoodSources;

	/**
	 * Armazena os Scout Bees e para posteriormente incluí-los como fonte de
	 * alimento
	 */
	private Set<FoodSource> scouts;

	/**
	 * Armazena as fontes abandonadas para posteriormente removê-las como fonte
	 * de alimento
	 */
	private Set<FoodSource> abandoned;

	/**
	 * Indica se na exploração modificaremos apenas uma feature por vez
	 * {@link PerturbationStrategy.CHANGE_ONE_FEATURE} ou através do parâmetro
	 * de perturbação MR {@link PerturbationStrategy.USE_MR}
	 */
	private PerturbationStrategy perturbation;

	/**
	 * Guarda quantas fontes de alimento o algoritmo consultou durante a busca
	 */
	private long states;

	private FileUtil fileUtil;

	private Classifier knnClassifier = new IBk();

	private Instances originalInstances;
	
	private HazelcastInstance hazelcastInstance;
	
	private IExecutorService executor;


	/**
	 * Construtor para seleção de features com a fontes as fontes de alimento
	 * sendo inicializadas uma por feature e utilizando o parâmetro de
	 * perturbação MR que define quantos parâmetros são moficados ao explorar a
	 * vizinhança
	 * 
	 * @param runtime
	 *            Número de execuções do ABC
	 * @param limit
	 *            Máximo número de vezes que será explorado uma vizinhança,
	 *            antes de abandonar a fonte de alimento
	 * @param mr
	 *            Parâmetro de perturbação para definir quantas features serão
	 *            incluídas ou excluídas durante a exploração. Se mr > 0
	 *            perturbation = PerturbationStrategy.USE_MR, se mr <=0
	 *            perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE;
	 * @param executor
	 *            Responsável pela classificação
	 * 
	 */
	public FeatureSelection(int runtime, int limit, double mr,
			String databaseName, Filter... filter) {
		this.runtime = runtime;
		this.limit = limit;
		this.mr = mr;
		if (mr > 0) {
			perturbation = PerturbationStrategy.USE_MR;
		} else {
			perturbation = PerturbationStrategy.CHANGE_ONE_FEATURE;
		}
		try {
			// carrega os dados do arquivo
			originalInstances = new Instances(new FileReader(
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
					filter[i].setInputFormat(originalInstances);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
				try {
					originalInstances = Filter.useFilter(originalInstances,
							filter[i]);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}

			}
		}
		this.featureSize = originalInstances.numAttributes() - 1;
	}

	/**
	 * Construtor para seleção de features com a fontes as fontes de alimento
	 * sendo inicializadas uma por feature e alterando apenas uma feature por
	 * vez ao explorar a vizinhança
	 * 
	 * @param runtime
	 *            Número de execuções do ABC
	 * @param limit
	 *            Máximo número de vezes que será explorado uma vizinhança,
	 *            antes de abandonar a fonte de alimento
	 * @param executor
	 *            Responsável pela classificação
	 */
	public FeatureSelection(int runtime, int limit, String databaseName,
			Filter... filter) {

		this(runtime, limit, 0, databaseName, filter);
	}

	/**
	 * Executa todo o processo de seleção
	 */
	public void execute() {
		hazelcastInstance = Hazelcast.newHazelcastInstance();
		visitedFoodSources = new ConcurrentSkipListSet<FoodSource>();
		abandoned = new ConcurrentSkipListSet<FoodSource>();

		fileUtil = FileUtil.newInstance();
		writer = fileUtil.getWriter();

		bestFitness = 0;

		states = 0;

		logFeatureSeletionInit(runtime, limit, mr, perturbation, featureSize);
		long time = System.currentTimeMillis();

		initializeFoodSources();
		for (int i = 0; i < runtime; i++) {
			scouts = new ConcurrentSkipListSet<FoodSource>();
			if (i == 0) {
				sendEmployedBees();
			}
			fileUtil.flush();
			sendOnlookerBees();
			fileUtil.flush();
			sendScoutBeesAndRemoveAbandonsFoodSource();
			fileUtil.flush();
		}
		time = (System.currentTimeMillis() - time) / 60000;
		logBestSolutionAndExecutionTime(time);
		executor.shutdown();
		Set<HazelcastInstance> hzInstances = Hazelcast.getAllHazelcastInstances();
		for (HazelcastInstance hzInstance : hzInstances) {
			hzInstance.shutdown();
		}
		hazelcastInstance.shutdown();
		states = 0;
	}

	/**
	 * Inicializa as fontes de alimento com o mesmo número de features, cada
	 * fonte tem selecionado apenas uma feature, a medida que o algoritmo é
	 * executado novas features são adicionadas ao conjunto
	 */
	private void initializeFoodSources() {

		System.out.println("initializeFoodSources");
		
		foodSources = hazelcastInstance.getSet("Food-Sources");
		executor = hazelcastInstance
				.getExecutorService("ABC-feature-selection");

		Set<Member> clusterMembers = hazelcastInstance.getCluster()
				.getMembers();
		
		int clusterSize = clusterMembers.size();

		int index = 0;

		int loopSize = featureSize / clusterSize;

		int rest = featureSize % clusterSize;

		for (int i = 0; i < loopSize; i++) {
			Queue<Future<FoodSource>> queueFuture = new LinkedList<Future<FoodSource>>();
			for (Member member : clusterMembers) {
				states++;
				boolean features[] = new boolean[featureSize];
				features[index++] = true;
				// calcula a qualidade do conjunto criado paralelizando em
				// featureSize threads

				System.out.print(index + " ");
				ClassifierExecutionCallable callable = new ClassifierExecutionCallable(
						features, originalInstances, knnClassifier);
				queueFuture.add(executor.submitToMember(callable, member));
			}
			Future<FoodSource> foodsFuture = null;
			while ((foodsFuture = queueFuture.poll()) != null) {
				// recupera os valores conforme são retornados da thread
				FoodSource foodSource = null;
				try {
					foodSource = foodsFuture.get();
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				} catch (ExecutionException e) {
					throw new RuntimeException(e);
				}
				// calcula a qualidade do conjunto criado
				double curFitness = foodSource.getFitness();
				// adicionamos fonte de alimento
				foodSources.add(foodSource);
				// armazena a melhor
				if (curFitness > bestFitness) {
					bestFoodSource = new FoodSource(foodSource);
					bestFitness = curFitness;
				}
			}
		}
		Queue<Future<FoodSource>> queueFuture = new LinkedList<Future<FoodSource>>();
		int i = 0;
		for (Member member : clusterMembers) {
			if (i >= rest) {
				break;
			}
			states++;
			boolean features[] = new boolean[featureSize];
			features[index++] = true;
			// calcula a qualidade do conjunto criado paralelizando em
			// featureSize threads
			System.out.print(index + " ");
			ClassifierExecutionCallable callable = new ClassifierExecutionCallable(
					features, originalInstances, knnClassifier);
			queueFuture.add(executor.submitToMember(callable, member));
			i++;
		}
		System.out.println("Get ");
		Future<FoodSource> foodsFuture = null;
		while ((foodsFuture = queueFuture.poll()) != null) {

			// recupera os valores conforme são retornados da thread
			FoodSource foodSource = null;
			try {
				foodSource = foodsFuture.get();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			} catch (ExecutionException e) {
				throw new RuntimeException(e);
			}
			// calcula a qualidade do conjunto criado
			double curFitness = foodSource.getFitness();
			// adicionamos fonte de alimento
			foodSources.add(foodSource);
			// armazena a melhor
			if (curFitness > bestFitness) {
				bestFoodSource = new FoodSource(foodSource);
				bestFitness = curFitness;
			}
		}
	}

	/**
	 * Envia as abelhas para explorar as fontes de alimento e sua vizinhança
	 */
	private void sendEmployedBees() {

		System.out.println("sendEmployedBees");

		int foodSourceListSize = foodSources.size();
		System.out.println(foodSourceListSize);

		executor = hazelcastInstance
				.getExecutorService("ABC-feature-selection");

		Set<Member> clusterMembers = hazelcastInstance.getCluster()
				.getMembers();
		int clusterSize = clusterMembers.size();

		int index = 0;

		int loopSize = foodSourceListSize / clusterSize;

		int rest = foodSourceListSize % clusterSize;

		if (foodSourceListSize != 0) {

			FoodSource[] sources = foodSources.toArray(new FoodSource[0]);

			for (int i = 0; i < loopSize; i++) {
				Queue<Future<BeeParallelExecutionResult>> queueFuture = new LinkedList<Future<BeeParallelExecutionResult>>();
				for (Member member : clusterMembers) {
					SendBeeCallable callable = new SendBeeCallable(
							sources[index++], originalInstances, knnClassifier,
							limit, mr);
					queueFuture.add(executor.submitToMember(callable, member));
					System.out.print(index + " ");
				}
				Future<BeeParallelExecutionResult> beeFuture = null;
				while ((beeFuture = queueFuture.poll()) != null) {
					try {
						BeeParallelExecutionResult result = beeFuture.get();
						foodSources.removeAll(result.getMarkedToRemoved());
						foodSources.addAll(result.getNeighbors());
						abandoned.addAll(result.getAbandoned());
						int abandonedSize = result.getAbandoned().size();
						for (int l = 0; l < abandonedSize; l++) {
							createScoutBee();
						}
						visitedFoodSources.addAll(result
								.getVisitedFoodSources());
						if (result.getBestFitness() > bestFitness
								|| (result.getBestFitness() == bestFitness && result
										.getBestFoodSource().getNrFeatures() < bestFoodSource
										.getNrFeatures())) {

							bestFoodSource = new FoodSource(
									result.getBestFoodSource());
							bestFitness = result.getBestFitness();
						}
					} catch (InterruptedException e) {
						throw new RuntimeException(e);
					} catch (ExecutionException e) {
						throw new RuntimeException(e);
					}

				}
			}
			Queue<Future<BeeParallelExecutionResult>> queueFuture = new LinkedList<Future<BeeParallelExecutionResult>>();
			int i = 0;
			for (Member member : clusterMembers) {
				if (i >= rest) {
					break;
				}
				SendBeeCallable callable = new SendBeeCallable(
						sources[index++], originalInstances, knnClassifier,
						limit, mr);
				queueFuture.add(executor.submitToMember(callable, member));
				System.out.print(index + " ");
				i++;
			}
			Future<BeeParallelExecutionResult> beeFuture = null;
			while ((beeFuture = queueFuture.poll()) != null) {
				try {
					BeeParallelExecutionResult result = beeFuture.get();
					foodSources.removeAll(result.getMarkedToRemoved());
					foodSources.addAll(result.getNeighbors());
					abandoned.addAll(result.getAbandoned());
					int abandonedSize = result.getAbandoned().size();
					for (int l = 0; l < abandonedSize; l++) {
						createScoutBee();
					}
					visitedFoodSources.addAll(result.getVisitedFoodSources());
					if (result.getBestFitness() > bestFitness
							|| (result.getBestFitness() == bestFitness && result
									.getBestFoodSource().getNrFeatures() < bestFoodSource
									.getNrFeatures())) {

						bestFoodSource = new FoodSource(
								result.getBestFoodSource());
						bestFitness = result.getBestFitness();
					}
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				} catch (ExecutionException e) {
					throw new RuntimeException(e);
				}

			}
		}
	}

	private void sendOnlookerBees() {

		System.out.println("sendOnlookerBees");

		int foodSourceListSize = foodSources.size();

		executor = hazelcastInstance
				.getExecutorService("ABC-feature-selection");

		Set<Member> clusterMembers = hazelcastInstance.getCluster()
				.getMembers();
		int clusterSize = clusterMembers.size();

		int index = 0;

		int loopSize = foodSourceListSize / clusterSize;

		int rest = foodSourceListSize % clusterSize;

		if (foodSourceListSize != 0) {

			Double min = Collections.min(foodSources).getFitness();
			Double range = Collections.max(foodSources).getFitness() - min;
			Random random = new Random();

			FoodSource[] sources = foodSources.toArray(new FoodSource[0]);

			for (int i = 0; i < loopSize; i++) {

				Queue<Future<BeeParallelExecutionResult>> queueFuture = new LinkedList<Future<BeeParallelExecutionResult>>();
				for (Member member : clusterMembers) {
					Double prob = (sources[index].getFitness() - min) / range;
					if (random.nextDouble() < prob) {
						SendBeeCallable callable = new SendBeeCallable(
								sources[index++], originalInstances,
								knnClassifier, limit, mr);
						queueFuture.add(executor.submitToMember(callable,
								member));
					} else {
						sources[index++].incrementLimit();
					}
					System.out.print(index + " ");
				}
				Future<BeeParallelExecutionResult> beeFuture = null;
				while ((beeFuture = queueFuture.poll()) != null) {
					try {
						BeeParallelExecutionResult result = beeFuture.get();
						foodSources.removeAll(result.getMarkedToRemoved());
						foodSources.addAll(result.getNeighbors());
						abandoned.addAll(result.getAbandoned());
						int abandonedSize = result.getAbandoned().size();
						for (int l = 0; l < abandonedSize; l++) {
							createScoutBee();
						}
						visitedFoodSources.addAll(result
								.getVisitedFoodSources());
						if (result.getBestFitness() > bestFitness
								|| (result.getBestFitness() == bestFitness && result
										.getBestFoodSource().getNrFeatures() < bestFoodSource
										.getNrFeatures())) {

							bestFoodSource = new FoodSource(
									result.getBestFoodSource());
							bestFitness = result.getBestFitness();
						}
					} catch (InterruptedException e) {
						throw new RuntimeException(e);
					} catch (ExecutionException e) {
						throw new RuntimeException(e);
					}
				}
			}
			int i = 0;
			Queue<Future<BeeParallelExecutionResult>> queueFuture = new LinkedList<Future<BeeParallelExecutionResult>>();
			for (Member member : clusterMembers) {
				if (i >= rest) {
					break;
				}

				Double prob = (sources[index].getFitness() - min) / range;
				if (random.nextDouble() < prob) {
					SendBeeCallable callable = new SendBeeCallable(
							sources[index++], originalInstances, knnClassifier,
							limit, mr);
					queueFuture.add(executor.submitToMember(callable, member));
				} else {
					sources[index++].incrementLimit();
				}
				System.out.print(index + " ");
				i++;
			}
			Future<BeeParallelExecutionResult> beeFuture = null;
			while ((beeFuture = queueFuture.poll()) != null) {
				try {
					BeeParallelExecutionResult result = beeFuture.get();
					foodSources.removeAll(result.getMarkedToRemoved());
					foodSources.addAll(result.getNeighbors());
					abandoned.addAll(result.getAbandoned());
					int abandonedSize = result.getAbandoned().size();
					for (int l = 0; l < abandonedSize; l++) {
						createScoutBee();
					}
					visitedFoodSources.addAll(result.getVisitedFoodSources());
					if (result.getBestFitness() > bestFitness
							|| (result.getBestFitness() == bestFitness && result
									.getBestFoodSource().getNrFeatures() < bestFoodSource
									.getNrFeatures())) {

						bestFoodSource = new FoodSource(
								result.getBestFoodSource());
						bestFitness = result.getBestFitness();
					}
				} catch (InterruptedException e) {
					throw new RuntimeException(e);
				} catch (ExecutionException e) {
					throw new RuntimeException(e);
				}
			}
		}
	}

	/**
	 * Cria uma nova fonte de alimento pelo scout
	 */
	private void createScoutBee() {
		boolean features[] = new boolean[featureSize];
		Random random = new Random();
		FoodSource foodSource = null;
		int nrFeatures = 0;
		// novas features
		for (int j = 0; j < featureSize; j++) {
			boolean inclusion = random.nextBoolean();
			if (inclusion) {
				nrFeatures++;
			}
			features[j] = inclusion;
		}

		double curFitness = calculateFitness(features);
		// cria uma nova fonte
		foodSource = new FoodSource(features, curFitness, nrFeatures);
		// se não existe ainda adiciona na lista de scout para ser usada
		// posteriormente
		if (!(foodSources.contains(foodSource)
				|| abandoned.contains(foodSource) || visitedFoodSources
					.contains(foodSource))) {
			states++;
			scouts.add(foodSource);
		}
	}

	/**
	 * Após enviar as employed e Onlookers bees remove todas as fontes
	 * abandonadas e adiciona as novas fontes encontradas pelos scout bees
	 */
	private void sendScoutBeesAndRemoveAbandonsFoodSource() {
		foodSources.removeAll(abandoned);
		foodSources.addAll(scouts);
	}

	/**
	 * Calcula a qualidade de uma fonte de alimento através da fonte selecionada
	 * 
	 * @param features
	 *            Features utilizadas para o cálculo da qualidade da fonte de
	 *            alimento
	 * @return valor de 0 a 100 que indica a qualidade da fonte
	 */
	private double calculateFitness(boolean features[]) {

		ClassifierExecutor executor = new KFoldClassifierExecutor(knnClassifier);
		// carrega as features
		executor.setOriginalInstances(originalInstances);
		executor.loadFeatures();
		// chama o classificador e rtorna a acurácia
		return executor.execute(features, KFOLD);
	}

	public void logFeatureSeletionInit(int runtime, int limit, double mr,
			PerturbationStrategy perturbation, int nrFeatures) {

		try {
			writer.write("Feature Selection START--------------------------------------------------------------------------------------------------------");

			writer.newLine();
			writer.write("Runtime [" + runtime + "], Limit [" + limit
					+ "], MR [" + mr + "], perturbation [" + perturbation);
			writer.newLine();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}

	public void logBestSolutionAndExecutionTime(long time) {
		try {
			writer.write("Best " + bestFoodSource);
			writer.newLine();
			writer.write("Executado em " + time + " percorrendo " + states
					+ " estados");
			writer.newLine();
			writer.write("Feature Selection END----------------------------------------------------------------------------------------------------------");
			writer.newLine();
		} catch (IOException e) {
			throw new RuntimeException();
		}
	}
}

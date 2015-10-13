package edu.umd.cs.aspect

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables
import util.DataOutputter;
import util.FoldUtils;
import util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics.BinaryClass;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.kernel.predicateconstraint.*


class Fineaspect_CV_setup{
	public static void main(String[] args)
	{
		for (int d = 1;d<11;++d)
		{
				new File(System.getProperty("user.home")+"output/crossvalidation_outputs/fold"+d).mkdir()
		}
		for(int i = 0; i <1; ++i)
		{
			Fineaspect_CV_setup a = new Fineaspect_CV_setup()
			a.pslmodel(i);
		}
	}

	void pslmodel(int cvSet)
	{
		int folds = 1 // number of folds
		double seedRatio = 1 // ratio of observed labels
		Random rand = new Random(0) // used to seed observed data
		double trainTestRatio = 0.9 // ratio of train to test splits (random)
		double filterRatio = 1.0 // ratio of documents to keep (throw away the rest)
		//targetSize = 3000 // target size of snowball sampler
		double explore = 0.001 // prob of random node in snowball sampler
		int numCategories = 7  // number of label categories
		ConfigManager cm = ConfigManager.getManager()
		ConfigBundle config = cm.getBundle("mooc-aspect-test")
		Logger log = LoggerFactory.getLogger(this.class)

		/* Uses H2 as a DataStore and stores it in a temp. directory by default */
		def defaultPath = System.getProperty("java.io.tmpdir")
		String dbpath = config.getString("dbpath", defaultPath + File.separator + "mooc-aspect-test")
		DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)
		String writefolder = "output/crossvalidation_outputs/"
		PSLModel m = new PSLModel(this, data)
		Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
		results.put(config, new ArrayList<DiscretePredictionStatistics>())

//Predicates

		m.add predicate: "relevance" , types: [ArgumentType.UniqueID]
		m.add predicate: "relevancepred" , types: [ArgumentType.UniqueID]
		m.add predicate: "seededLDA_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "seededLDA_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "seededLDA_3" , types: [ArgumentType.UniqueID]

		m.add predicate: "post" , types: [ArgumentType.UniqueID]
		m.add predicate: "postfinecategory1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID,ArgumentType.UniqueID]
		m.add predicate: "coarsepred0" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsepred1" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsepred2" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsepred3" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsepred4" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsepred5" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsepred6" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		m.add predicate: "finepred0" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred1" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred2" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred3" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred4" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred5" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred6" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred7" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred8" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred9" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred10" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred11" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred12" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred13" , types: [ArgumentType.UniqueID]
		m.add predicate: "finepred14" , types: [ArgumentType.UniqueID]
		m.add predicate: "child", types: [ArgumentType.String, ArgumentType.String]

		/*
		 * Adding rules
		 */
		

		m.add rule : (~relevancepred(P) & post(P)) >> finetopic(P, data.getUniqueID(0),data.getUniqueID(0)), weight : 5, squared:true
		m.add rule :  (finepred0(P) & ~relevancepred(P)) >> finetopic(P,data.getUniqueID(0),data.getUniqueID(0)), weight : 5, squared:true
		m.add rule :  (finepred1(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(1)), weight : 5, squared:true
		m.add rule :  (finepred2(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(2)), weight : 5, squared:true
		m.add rule :  (finepred3(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(3)), weight : 5, squared:true
		m.add rule :  (finepred4(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(4)), weight : 5, squared:true
		m.add rule :  (finepred5(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(5)), weight : 5, squared:true
		m.add rule :  (finepred6(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(6)), weight : 5, squared:true
		m.add rule :  (finepred7(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(7)), weight : 5, squared:true
		m.add rule :  (finepred8(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(8)), weight : 5, squared:true
		m.add rule :  (finepred9(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(9)), weight : 5, squared:true
		m.add rule :  (finepred10(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(10)), weight : 5, squared:true
		m.add rule :  (finepred11(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(11)), weight : 5, squared:true
		m.add rule :  (finepred12(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(12)), weight : 5, squared:true
		m.add rule :  (finepred13(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(13)), weight : 5, squared:true
		m.add rule :  (finepred14(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(14)), weight : 5, squared:true
		
		m.add rule :  (coarsepred0(P)) >> finetopic(P,data.getUniqueID(0),data.getUniqueID(0)), weight : 5, squared:true
		m.add rule :  (coarsepred1(P) & child("1","1")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(1)), weight : 10, squared:true
		m.add rule :  (coarsepred1(P) & child("1","2")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(2)), weight : 10, squared:true
		
		m.add rule :  (coarsepred1(P) & child("1","3")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(3)), weight : 10, squared:true
		m.add rule :  (coarsepred1(P) & child("1","4")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(4)), weight : 5, squared:true
		m.add rule :  (coarsepred1(P) & child("1","5")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(5)), weight : 5, squared:true
		m.add rule :  (coarsepred1(P) & child("1","6")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(6)), weight : 5, squared:true
		m.add rule :  (coarsepred1(P) & child("1","7")) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(7)), weight : 5, squared:true
		m.add rule :  (coarsepred2(P) & child("2","8")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(8)), weight : 10, squared:true
		m.add rule :  (coarsepred2(P) & child("2","9")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(9)), weight : 10, squared:true
		m.add rule :  (coarsepred2(P) & child("2","10")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(10)), weight : 10, squared:true
		m.add rule :  (coarsepred2(P) & child("2","11")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(11)), weight : 5, squared:true
		m.add rule :  (coarsepred2(P) & child("2","12")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(12)), weight : 5, squared:true
		m.add rule :  (coarsepred2(P) & child("2","13")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(13)), weight : 5, squared:true
		m.add rule :  (coarsepred2(P) & child("2","14")) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(14)), weight : 5, squared:true
		
		m.add rule : (coarsepred1(P)) >> (finetopic(P,data.getUniqueID(1),data.getUniqueID(1)) | finetopic(P,data.getUniqueID(1),data.getUniqueID(2)) |
		finetopic(P,data.getUniqueID(1),data.getUniqueID(3)) | finetopic(P,data.getUniqueID(1),data.getUniqueID(4)) |finetopic(P,data.getUniqueID(1),data.getUniqueID(5)) |
		finetopic(P,data.getUniqueID(1),data.getUniqueID(6)) | finetopic(P,data.getUniqueID(1),data.getUniqueID(7))), constraint:true
	
		m.add rule : (coarsepred2(P)) >> (finetopic(P,data.getUniqueID(2),data.getUniqueID(8)) | finetopic(P,data.getUniqueID(2),data.getUniqueID(11)) |
		finetopic(P,data.getUniqueID(2),data.getUniqueID(9)) | finetopic(P,data.getUniqueID(2),data.getUniqueID(10)) |finetopic(P,data.getUniqueID(2),data.getUniqueID(12)) |
		finetopic(P,data.getUniqueID(2),data.getUniqueID(13)) | finetopic(P,data.getUniqueID(2),data.getUniqueID(14))), constraint:true
	
	
		m.addKernel(new DomainRangeConstraintKernelAll(finetopic,DomainRangeConstraintType.Functional))
		

		List<Partition> trainPartition = new ArrayList<Partition>(folds)
		List<Partition> trainLabelsPartition = new ArrayList<Partition>(folds)
		List<Partition> testDataPartition = new ArrayList<Partition>(folds)
		List<Partition> testLabelsPartition = new ArrayList<Partition>(folds)

		/*
		 * Initialize partitions for all cross validation sets
		 */
		for(int initset =0 ;initset<10;++initset)
		{
			trainPartition.add(initset, new Partition(initset))
			trainLabelsPartition.add(initset, new Partition(initset + folds))
			testDataPartition.add(initset, new Partition(initset + 2*folds))
			testLabelsPartition.add(initset, new Partition(initset + 3*folds))
		}


		/*
		 * Set the folder to write into
		 */
		Integer folder = (cvSet+10)%10;
		if (folder ==0) folder = 10
//		String filename1 = writefolder+"fold"+folder+"/possentiment.csv"
//		String filename2 = writefolder+"fold"+folder+"/negsentiment.csv"
		File file1 = new File(writefolder+"results.txt");
//		File file2 = new File(filename2);
		File modelfile = new File(writefolder + "model.txt")
		
		Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			weights.put(k, k.getWeight());
		/*
		 * Train data partition, each partition has 9 folders, one kept aside for testing...
		 *
		 * loading the predicates from the data files into the trainPartition
		 */
		String filename
		Integer trainSet
		int countloadchild=0
		for (trainSet = 1 ; trainSet<=9;++trainSet)
		{
			Integer dirToUse = 0;
			dirToUse = (cvSet+trainSet)%10
			if(dirToUse==0) dirToUse = 10;
			String filepath = ""
			filename = 'data'+java.io.File.separator+'fold'+dirToUse+java.io.File.separator;
			for (Predicate p : [seededLDA_1, seededLDA_2, seededLDA_3, relevancepred])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
					filepath,"\t");
			}
			for (Predicate p : [post, postfinecategory1])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedData(data.getInserter(p, trainPartition.get(cvSet)),
					filepath,"\t");

			}
			if(countloadchild==0){
				String childfile = 'data'+java.io.File.separator+"child.txt";
				println "\t\t\tREADING child" +" from "+childfile+"...";
				InserterUtils.loadDelimitedData(data.getInserter(child, trainPartition.get(cvSet)),
						childfile,"\t");
					countloadchild = 1
			}
			
			for (Predicate p : [coarsepred0, coarsepred1, coarsepred2, coarsepred3, coarsepred4, coarsepred5, coarsepred6])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
					filepath,"\t");
			}
			for (Predicate p : [finepred0, finepred1, finepred2, finepred3, finepred4, finepred5, finepred6,finepred7,finepred8,finepred9,finepred10,finepred11,finepred12,finepred13,finepred14])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
					filepath,"\t");
			}
			
			/*
			  * Load in the ground truth labels for train partition
			  */
			String finetopicpath = filename+"finetopic.txt"
			 InserterUtils.loadDelimitedData(data.getInserter(finetopic, trainLabelsPartition.get(cvSet)),
				 finetopicpath);
		  }
	
		/*
		 * For test data partition - it needs only one fold in each partition.... Start with 10,1,2,3.... so on.
		 */
		Integer testSet = 0;
		testSet = (cvSet+10)%10
		if(testSet==0) testSet = 10;
		countloadchild = 0
		String filepath = ""
		filename = 'data'+java.io.File.separator+'fold'+testSet+java.io.File.separator;
		for (Predicate p : [seededLDA_1, seededLDA_2, seededLDA_3, relevancepred])
		{
			filepath = filename+p.getName().toString().toLowerCase()+".txt"
			println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
			InserterUtils.loadDelimitedDataTruth(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
		}
		for (Predicate p : [post, postfinecategory1])
		{
			filepath = filename+p.getName().toString().toLowerCase()+".txt"
			println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
			InserterUtils.loadDelimitedData(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
		}
		if(countloadchild==0)
		{
			String childfile = 'data'+java.io.File.separator+"child.txt";
			println "\t\t\tREADING child" +" from "+childfile+"... into test";
			InserterUtils.loadDelimitedData(data.getInserter(child, testDataPartition.get(cvSet)),childfile,"\t");
				countloadchild = 1
		}
		for (Predicate p : [coarsepred0, coarsepred1, coarsepred2, coarsepred3, coarsepred4, coarsepred5, coarsepred6])
		{
			filepath = filename+p.getName().toString().toLowerCase()+".txt"
			println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
			InserterUtils.loadDelimitedDataTruth(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
		}
		for (Predicate p : [finepred0, finepred1, finepred2, finepred3, finepred4, finepred5, finepred6,finepred7,finepred8,finepred9,finepred10,finepred11,finepred12,finepred13,finepred14])
		{
			filepath = filename+p.getName().toString().toLowerCase()+".txt"
			println "\t\t\tREADING " + p.getName() +" from "+filepath+"... into test";
			InserterUtils.loadDelimitedDataTruth(data.getInserter(p, testDataPartition.get(cvSet)),filepath,"\t");
		}
		/*
		  * Load in the ground truth labels for train partition
		  */
		String finetopicpath = filename+"finetopic.txt"
		 InserterUtils.loadDelimitedData(data.getInserter(finetopic, testLabelsPartition.get(cvSet)), finetopicpath);
		 

		 
		 Set toClose = [seededLDA_1, seededLDA_2, seededLDA_3, relevancepred,child, post, postfinecategory1,coarsepred0, coarsepred1, coarsepred2, coarsepred3, coarsepred4, coarsepred5, coarsepred6,
			 finepred0, finepred1, finepred2, finepred3, finepred4, finepred5, finepred6,finepred7,finepred8,finepred9,finepred10,finepred11,finepred12,finepred13,finepred14] as Set;

		 Database trainDB = data.getDatabase(trainPartition.get(cvSet), toClose as Set);
		 Database trainLabelsDB = data.getDatabase(trainLabelsPartition.get(cvSet), [finetopic] as Set);
		 
		 ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory1))
		 println "groundings for all "+ allGroundings.size();
		for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = trainDB.getAtom(finetopic, grounding);
			atom1.setValue(0.0);
			atom1.commitToDB();
		}
	 
		 
		 MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, trainDB, trainLabelsDB, config);
		 weightLearning.learn();
		 weightLearning.close();
		 modelfile.append(m)
		 modelfile.append("\n")
		 
		 
		 Database testDB = data.getDatabase(testDataPartition.get(cvSet), toClose)
		 Database testLabelsDB = data.getDatabase(testLabelsPartition.get(cvSet), [finetopic] as Set);

		 
		 
	  	allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory1))
		println "groundings for all "+ allGroundings.size();
	 	for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = testDB.getAtom(finetopic, grounding);
			atom1.setValue(0.0);
			atom1.commitToDB();
	 	}
		
			MPEInference mpe = new MPEInference(m, testDB, config)
			FullInferenceResult result = mpe.mpeInference()
			System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

			for (GroundAtom atom : Queries.getAllAtoms(testDB, finetopic)){
				file1.append( atom.toString() + "\t" + atom.getValue()+"\n");
			}

		 
			
			allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
			System.out.println("printing no of atoms in test db " +allGroundings.size());
			testLabelsDB.close()
			def groundTruthDB = data.getDatabase(testLabelsPartition.get(cvSet), [finetopic] as Set)
			DataOutputter.outputPredicate("output" + "/groundTruth" + cvSet + ".node" , groundTruthDB, finetopic, "\t", false, "nodeid\tlabel")
			
			allGroundings = groundTruthDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
			System.out.println("printing no of atoms in groundtruth db " +allGroundings.size());
			int totalTestExamples = allGroundings.size()
			DataOutputter.outputClassificationPredictions("output" + "/results" + config.getString("name", "") + cvSet + ".txt",
				testDB, finetopic, "\t")
			
			allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
			for(int k = 0;k<allGroundings.size();++k){
				GroundTerm [] grounding = allGroundings.get(k)
				System.out.println(grounding);
			}
			HashMap <GroundTerm,Integer> map = new HashMap<GroundTerm, Integer>()
			for(int k = 0;k<allGroundings.size();++k){
				GroundTerm [] grounding = allGroundings.get(k)
				String index = grounding[2].toString()
				if(!map.containsKey(grounding[2])){
					map.put(grounding[2], Integer.parseInt(index))}
			}
			def comparator = new MulticlassPredictionComparator(testDB)
			comparator.setBaseline(groundTruthDB)
			comparator.setResultFilter(new MaxValueFilter(finetopic, 1))
			//comparator.setThreshold(Double.MIN_VALUE)
			MulticlassPredictionStatistics stats = comparator.compare(finetopic, map,2)
			println stats.getAccuracy()
			
			
			
//			def comparator = new DiscretePredictionComparator(testDB)
//
//			//groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [coarsetopic] as Set)
//			comparator.setBaseline(groundTruthDB)
//			comparator.setResultFilter(new MaxValueFilter(finetopic, 1))
//			comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero
//
//			System.out.println("totalTestExamples " + totalTestExamples)
//			DiscretePredictionStatistics stats = comparator.compare(finetopic, totalTestExamples)
//			System.out.println("F1 score positive " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
//
//			def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
//			System.out.println("Method " + config.getString("name", "") + ", fold " + cvSet +", acc " + stats.getAccuracy() +
//					", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
//					", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
//					", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
//
//
//			results.get(config).add(cvSet, stats)
//
			
			groundTruthDB.close()
			testDB.close()
			trainDB.close()
			
		 
	}
}

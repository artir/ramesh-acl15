package edu.umd.cs.aspect

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables
import util.DataOutputter;
import util.ExperimentConfigGenerator;
import util.FoldUtils;
import util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.application.learning.weight.em.HardEM
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


class HierarchicalUnsupervised{
	public static void main(String[] args)
	{
		for (int d = 1;d<11;++d)
		{
				new File(System.getProperty("user.home")+"output/crossvalidation_outputs/fold"+d).mkdir()
		}
		for(int i = 0; i <10; ++i)
		{
			HierarchicalUnsupervised a = new HierarchicalUnsupervised()
			a.pslmodel(i);
		}
	}

	void pslmodel(int cvSet)
	{
		int folds = 10 // number of folds
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

		m.add predicate: "post" , types: [ArgumentType.UniqueID]
		m.add predicate: "postfinecategory1" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
		m.add predicate: "coarsecategory" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
		
		m.add predicate: "coarseseeded_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_3" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_4" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_5" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_6" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_7" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarseseeded_8" , types: [ArgumentType.UniqueID]
		
		m.add predicate: "coarsetopic_1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic_1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		
		//fineseeded
		m.add predicate: "fineseeded_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_3" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_4" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_5" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_6" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_7" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_8" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_9" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_10" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_11" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_12" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_13" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_14" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_15" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_16" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_17" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_18" , types: [ArgumentType.UniqueID]

		m.add predicate: "child", types: [ArgumentType.String, ArgumentType.String]
		m.add predicate: "lecture_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "quiz_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine1_neutral_sum" , types: [ArgumentType.UniqueID]

		/*
		 * Adding rules
		 */
		
		
		/*
		 * coarse seededlda to coarsetopic
		 */
		
		m.add rule :  (coarseseeded_1(P)) >> coarsetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
		m.add rule :  (coarseseeded_2(P)) >> coarsetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
		m.add rule :  (coarseseeded_3(P)) >> coarsetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
		m.add rule :  (coarseseeded_4(P)) >> coarsetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
		m.add rule :  (coarseseeded_5(P)) >> coarsetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true
		m.add rule :  (coarseseeded_6(P)) >> coarsetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true
		m.add rule :  (coarseseeded_7(P)) >> coarsetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true
		m.add rule :  (coarseseeded_8(P)) >> coarsetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true

		m.add rule :  (lecture_sum(P)) >> coarsetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
		m.add rule :  (~lecture_sum(P) & post(P)) >> ~coarsetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
		
		m.add rule :  (quiz_sum(P)) >> coarsetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true
		m.add rule :  (~quiz_sum(P) & post(P)) >> ~coarsetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true

		/*
		 * Fineseeded to finetopic, was weight 5 initially
		 */
		 
		 m.add rule : (fineseeded_1(P))	>>	finetopic_1(P, data.getUniqueID(2)), weight : 10, squared:true
		 m.add rule : (fineseeded_2(P))	>>	finetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
		 m.add rule : (fineseeded_3(P)) >> finetopic_1(P, data.getUniqueID(5)), weight : 10, squared:true
		 m.add rule : (fineseeded_4(P)) >> finetopic_1(P, data.getUniqueID(6)), weight : 10, squared:true
		 m.add rule : (fineseeded_5(P)) >> finetopic_1(P, data.getUniqueID(3)), weight : 10, squared:true
		 m.add rule : (fineseeded_6(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		 m.add rule : (fineseeded_7(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 10, squared:true
		 m.add rule : (fineseeded_8(P)) >> finetopic_1(P, data.getUniqueID(11)), weight :10 , squared:true
		 m.add rule : (fineseeded_9(P)) >> finetopic_1(P, data.getUniqueID(10)), weight : 10, squared:true
		 m.add rule : (fineseeded_10(P)) >> finetopic_1(P, data.getUniqueID(8)), weight : 10, squared:true
		 m.add rule : (fineseeded_11(P)) >> finetopic_1(P, data.getUniqueID(12)), weight : 10, squared:true
		 m.add rule : (fineseeded_12(P)) >> finetopic_1(P, data.getUniqueID(14)), weight : 10, squared:true
		 m.add rule : (fineseeded_13(P)) >> finetopic_1(P, data.getUniqueID(13)), weight : 10, squared:true
		 m.add rule : (fineseeded_14(P)) >> finetopic_1(P, data.getUniqueID(15)), weight : 10, squared:true
		 m.add rule : (fineseeded_15(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		 m.add rule : (fineseeded_16(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		 m.add rule : (fineseeded_17(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		 m.add rule : (fineseeded_18(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		 m.add rule : (fineseeded_15(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		 m.add rule : (fineseeded_16(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		 m.add rule : (fineseeded_17(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		 m.add rule : (fineseeded_18(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		
		
		/*
		 * fine seeded to coarsetopic
		 */
		
//		video,problem,download,play,player,watch,speed,length,long,fast,slow,render,qualiti
//		volum,low,headphon,sound,audio,hear,maximum,troubl,qualiti,high,loud,heard
//		professor,fast,speak,pace,follow,speed,slow,accent,absorb,quick,slower,slowli
//		transcript,subtitl,slide,note,lectur,difficult,pdf
//		avail,upload,time,ahead,nowher,find,post,access,error,issu,unabl,found,locat,receiv,open,broken,link,bad,access,deni,accessdeni,view,permiss
//		typo,error,mistak,wrong,right,incorrect,mistaken
//		question,challeng,difficulti,difficult,understand,typo,error,mistak,mistaken,quiz,assignment
//		submiss,submit,quiz,error,unabl,submission_id,submitbutton,resubmit,submitt
//		answer,question,answer,grade,assignment,quiz,respons,mark,wrong,score,grad
//		due,deadlin,miss,extend,late,dead,line
//		certif,score,signatur,statement,final,course,pass,receiv,coursera,accomplish,fail
//		supplement,book,note,wiki,packag,code,slide,read,buy
//		schedul,per,fall,behind,calendar,week,date,start
//		hi,i,am,name,mood,coursera,course,introduction,stud,group,everyon,student
//		android,develop,eclips,sdk,softwar,hardware,accuser,html,platform,environ,lab,ide,java,app,sourc,junit,test,onclick,virtual,launch,devic,mobil,jdk
//		protein,food,gene,vitamin,diet,neanderth,evolut,sequenc,chromosom,genet,speci,peopl,popul,evolv,mutat,ancestri
//		compani,product,industri,innovativeidea,entrepreneur,strategi,decision,
//		dirupt,technolog,market
		
		 
		 //Was all weight 5 initially
		m.add rule : (fineseeded_1(P))	>>	coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
		m.add rule : (fineseeded_2(P))	>>	coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
		m.add rule : (fineseeded_3(P)) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
		m.add rule : (fineseeded_4(P)) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
		m.add rule : (fineseeded_5(P)) >> coarsetopic_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_6(P)) >> coarsetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
//		m.add rule : (fineseeded_7(P)) >> coarsetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		m.add rule : (fineseeded_8(P)) >> coarsetopic_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_9(P)) >> coarsetopic_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_10(P)) >> coarsetopic_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_11(P)) >> coarsetopic_1(P, data.getUniqueID(3)), weight : 10, squared:true
		m.add rule : (fineseeded_12(P)) >> coarsetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		m.add rule : (fineseeded_13(P)) >> coarsetopic_1(P, data.getUniqueID(5)), weight : 10, squared:true
		m.add rule : (fineseeded_14(P)) >> coarsetopic_1(P, data.getUniqueID(6)), weight : 10, squared:true
//		m.add rule : (fineseeded_15(P)) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
//		m.add rule : (fineseeded_16(P)) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
//		m.add rule : (fineseeded_17(P)) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
//		m.add rule : (fineseeded_18(P)) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
				
		//		data_labels = {
		//			"lecture-audio" :	1,
		//			"lecture-video" :	2,
		//			"lecture-availability" :	3,
		//			"lecture-content" :	4,
		//			"lecture-lecturer" :	5,
		//			"lecture-subtitles" :	6,		
		//			"quiz-availability"	: 7,
		//			"quiz-deadlines"	: 8,
		//			"quiz-content"	: 9,
		//			"quiz-grading"	: 10,
		//			"quiz-submission"	: 11
		//			"certificate"	: 12,
		//			"schedule"	: 13,
		//			"supplements" : 14,
		//			"social"	: 15
		//			}
		
		
//		m.add rule : (finetopic_1(P, data.getUniqueID(1)) & finetopic_1(P, data.getUniqueID(2)) 
//			& finetopic_1(P, data.getUniqueID(3)) & finetopic_1(P, data.getUniqueID(4)) 
//			& finetopic_1(P, data.getUniqueID(5)) & 
//			finetopic_1(P, data.getUniqueID(6))) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
//
//		m.add rule : (finetopic_1(P, data.getUniqueID(7)) & finetopic_1(P, data.getUniqueID(8))
//			& finetopic_1(P, data.getUniqueID(9)) & finetopic_1(P, data.getUniqueID(10))
//			& finetopic_1(P, data.getUniqueID(11))) >> coarsetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true

//		m.add rule : (fine1_neutral_sum(P))>>coarsetopic_1(P, data.getUniqueID(6)), weight : 10, squared:true
		
		/*
		 * Fineseeded and coarsetopic - > fine topic, was all weight 5
		 */
		m.add rule : (fineseeded_1(P) & coarsetopic_1(P,data.getUniqueID(1)))	>>	finetopic_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_2(P) & coarsetopic_1(P,data.getUniqueID(1)))	>>	finetopic_1(P, data.getUniqueID(1)), weight : 10, squared:true
		m.add rule : (fineseeded_3(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(5)), weight : 10, squared:true
		m.add rule : (fineseeded_4(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(6)), weight : 10, squared:true
		m.add rule : (fineseeded_5(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(3)), weight : 10, squared:true
		m.add rule : (fineseeded_6(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		m.add rule : (fineseeded_7(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(9)), weight : 10, squared:true
		m.add rule : (fineseeded_8(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(11)), weight : 10, squared:true
		m.add rule : (fineseeded_9(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(10)), weight : 10, squared:true
		m.add rule : (fineseeded_10(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(8)), weight : 10, squared:true
		m.add rule : (fineseeded_11(P) & coarsetopic_1(P,data.getUniqueID(3))) >> finetopic_1(P, data.getUniqueID(12)), weight : 10, squared:true
		m.add rule : (fineseeded_12(P) & coarsetopic_1(P,data.getUniqueID(4))) >> finetopic_1(P, data.getUniqueID(14)), weight : 10, squared:true
		m.add rule : (fineseeded_13(P) & coarsetopic_1(P,data.getUniqueID(5))) >> finetopic_1(P, data.getUniqueID(13)), weight : 10, squared:true
		m.add rule : (fineseeded_14(P) & coarsetopic_1(P,data.getUniqueID(6))) >> finetopic_1(P, data.getUniqueID(15)), weight : 10, squared:true
		m.add rule : (fineseeded_15(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		m.add rule : (fineseeded_16(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		m.add rule : (fineseeded_17(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		m.add rule : (fineseeded_18(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P, data.getUniqueID(4)), weight : 10, squared:true
		m.add rule : (fineseeded_15(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(9)), weight : 10, squared:true
		m.add rule : (fineseeded_16(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(9)), weight : 10, squared:true
		m.add rule : (fineseeded_17(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(9)), weight : 10, squared:true
		m.add rule : (fineseeded_18(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P, data.getUniqueID(9)), weight : 10, squared:true
		/*
		 * Mapping Coarse topic to finetopics
		 */
//		m.add rule : coarsetopic_1(P,data.getUniqueID(1)) >> (finetopic_1(P,data.getUniqueID(1)) | finetopic_1(P,data.getUniqueID(2)) |
//				finetopic_1(P,data.getUniqueID(3)) | finetopic_1(P,data.getUniqueID(4)) |finetopic_1(P,data.getUniqueID(5)) |
//				finetopic_1(P,data.getUniqueID(6))), constraint:true
//			
//		m.add rule : coarsetopic_1(P,data.getUniqueID(2)) >> (finetopic_1(P,data.getUniqueID(7)) | finetopic_1(P,data.getUniqueID(8)) |
//				finetopic_1(P,data.getUniqueID(9)) | finetopic_1(P,data.getUniqueID(10)) |finetopic_1(P,data.getUniqueID(11))), constraint:true

		/*
		 * Mapping lecture sum to all lecture finetopics
		 * Mapping quiz sum to all quiz finetopics
		 */
			
//		m.add rule :  (lecture_sum(P)) >> (finetopic_1(P,data.getUniqueID(1)) | finetopic_1(P,data.getUniqueID(2)) |
//				finetopic_1(P,data.getUniqueID(3)) | finetopic_1(P,data.getUniqueID(4)) |finetopic_1(P,data.getUniqueID(5)) |
//				finetopic_1(P,data.getUniqueID(6))), weight : 10, squared:true
//					
//		m.add rule :  (quiz_sum(P)) >> (finetopic_1(P,data.getUniqueID(7)) | finetopic_1(P,data.getUniqueID(8)) |
//				finetopic_1(P,data.getUniqueID(9)) | finetopic_1(P,data.getUniqueID(10)) |
//				finetopic_1(P,data.getUniqueID(11))), weight : 10, squared:true
	
			
//			lectur,video,download,play,player,watch,speed,length,long,fast,slow,render,qualiti - 2
//			lectur,volum,low,headphon,sound,audio,hear,maximum,troubl,qualiti,high,loud,heard - 1
//			lectur,professor,fast,speak,pace,follow,speed,slow,accent,absorb,quick,slower,slowli - 5
//			lectur,transcript,subtitl,slide,note,lectur,difficult,pdf - 6
//			quiz,quizz,assignment,question,midterm,exam     - 9
//			submiss,submit,quiz,error,unabl,submission_id,submitbutton,resubmit,submitt - 11
//			answer,question,answer,grade,assignment,quiz,respons,mark,wrong,score,grad   - 10
//			quiz,exam,midterm,due,deadlin,miss,extend,late,dead,line - 8

		/*
		 * Coarse seededlda to finetopic	
		 */

		/*
		 * This hurt hte results for finetopic
		 */
//		m.add rule :  (coarseseeded_1(P)) >> finetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_2(P)) >> finetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_3(P)) >> finetopic_1(P,data.getUniqueID(5)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_4(P)) >> finetopic_1(P,data.getUniqueID(6)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_5(P)) >> finetopic_1(P,data.getUniqueID(9)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_6(P)) >> finetopic_1(P,data.getUniqueID(11)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_7(P)) >> finetopic_1(P,data.getUniqueID(10)), weight : 10, squared:true
//		m.add rule :  (coarseseeded_8(P)) >> finetopic_1(P,data.getUniqueID(8)), weight : 10, squared:true
	
		/*
		 * Coarse seededlda & coarsetopic => finetopic
		 */
		/*
		 * This hurt as well. 
		 */
//		m.add rule :(coarseseeded_1(P) & coarsetopic_1(P,data.getUniqueID(1)) ) >> finetopic_1(P,data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule :(coarseseeded_2(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P,data.getUniqueID(1)), weight : 10, squared:true
//		m.add rule :(coarseseeded_3(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P,data.getUniqueID(5)), weight : 10, squared:true
//		m.add rule :(coarseseeded_4(P) & coarsetopic_1(P,data.getUniqueID(1))) >> finetopic_1(P,data.getUniqueID(6)), weight : 10, squared:true
//		m.add rule :(coarseseeded_5(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P,data.getUniqueID(9)), weight : 10, squared:true
//		m.add rule :(coarseseeded_6(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P,data.getUniqueID(11)), weight : 10, squared:true
//		m.add rule :(coarseseeded_7(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P,data.getUniqueID(10)), weight : 10, squared:true
//		m.add rule :(coarseseeded_8(P) & coarsetopic_1(P,data.getUniqueID(2))) >> finetopic_1(P,data.getUniqueID(8)), weight : 10, squared:true

		
//		m.addKernel(new DomainRangeConstraintKernel(finetopic_1,DomainRangeConstraintType.Functional))
//		m.addKernel(new DomainRangeConstraintKernel(coarsetopic_1,DomainRangeConstraintType.Functional))
		

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
		for (trainSet = 1 ; trainSet<=10;++trainSet)
		{
			Integer dirToUse = 0;
			dirToUse = (cvSet+trainSet)%10
			if(dirToUse==0) dirToUse = 10;
			String filepath = ""
			filename = 'data/4144folds_new'+java.io.File.separator+'fold'+dirToUse+java.io.File.separator;
			for (Predicate p : [coarseseeded_1, coarseseeded_2,coarseseeded_3,coarseseeded_4,
				coarseseeded_5,coarseseeded_6,coarseseeded_7,coarseseeded_8,fineseeded_1,fineseeded_2,fineseeded_3,
				fineseeded_4,fineseeded_5,fineseeded_6,fineseeded_7,fineseeded_8,fineseeded_9,fineseeded_10,fineseeded_11,
				fineseeded_12,fineseeded_13,fineseeded_14,fineseeded_15,fineseeded_16,fineseeded_17,fineseeded_18,
				lecture_sum, quiz_sum,fine1_neutral_sum])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
					filepath,"\t");
			}
			for (Predicate p : [post, postfinecategory1,coarsecategory])
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
			
	

			
			/*
			  * Load in the ground truth labels for train partition
			  */
			
			String finetopic_1path = filename+"finetopic_1.txt"
			InserterUtils.loadDelimitedData(data.getInserter(finetopic_1, trainLabelsPartition.get(cvSet)),
					finetopic_1path);
			String sentiment_1path = filename+"coarsetopic_1.txt"
			InserterUtils.loadDelimitedData(data.getInserter(coarsetopic_1, trainLabelsPartition.get(cvSet)),
						sentiment_1path);

		}

		 
		 Set toClose = [child, post, postfinecategory1,coarseseeded_1, coarseseeded_2,coarseseeded_3,coarseseeded_4,
				coarseseeded_5,coarseseeded_6,coarseseeded_7,coarseseeded_8,fineseeded_1,fineseeded_2,fineseeded_3,
				fineseeded_4,fineseeded_5,fineseeded_6,fineseeded_7,fineseeded_8,fineseeded_9,fineseeded_10,fineseeded_11,
				fineseeded_12,fineseeded_13,fineseeded_14,fineseeded_15,fineseeded_16,fineseeded_17,fineseeded_18,lecture_sum,
				quiz_sum,coarsecategory,fine1_neutral_sum] as Set;

		 Database trainDB = data.getDatabase(trainPartition.get(cvSet), toClose as Set);
		 Database trainLabelsDB = data.getDatabase(trainLabelsPartition.get(cvSet), [finetopic_1, coarsetopic_1] as Set);
		 
		 ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory1))
		 println "groundings for all "+ allGroundings.size();
		for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = trainDB.getAtom(finetopic_1, grounding);
			atom1.setValue(0.0);
			atom1.commitToDB();
		}
		
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(coarsecategory))
		println "groundings for all "+ allGroundings.size();
	   for (int j = 0; j < allGroundings.size(); j++) {
		   GroundTerm [] grounding = allGroundings.get(j)
		   RandomVariableAtom atom1 = trainDB.getAtom(coarsetopic_1, grounding);
		   atom1.setValue(0.0);
		   atom1.commitToDB();
	   }

	 
		 
		 MPEInference mpe = new MPEInference(m, trainDB, config)
		 FullInferenceResult result = mpe.mpeInference()
		 System.out.println("Objective: " + result.getTotalWeightedIncompatibility())
		 modelfile.append(m)
		 modelfile.append("\n")
//		 groundTruthDB = data.getDatabase(trainLabelsPartition.get(cvSet), [coarsetopic_1, finetopic_1] as Set)
	
		 
		 allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(coarsetopic_1))
		 for(int k = 0;k<allGroundings.size();++k){
			 GroundTerm [] grounding = allGroundings.get(k)
			 System.out.println(grounding);
		 }
			HashMap <GroundTerm,Integer> map = new HashMap<GroundTerm, Integer>()
			for(int k = 0;k<allGroundings.size();++k){
				GroundTerm [] grounding = allGroundings.get(k)
				String index = grounding[1].toString()
				if(!map.containsKey(grounding[1])){
					map.put(grounding[1], Integer.parseInt(index))}
			}
			
			
			
			def comparator = new MulticlassPredictionComparator(trainDB)
			comparator.setBaseline(trainLabelsDB)
			
			
			comparator.setResultFilter(new MaxValueFilter(coarsetopic_1, 1))
			//comparator.setThreshold(Double.MIN_VALUE)
			MulticlassPredictionStatistics stats = comparator.compare(coarsetopic_1, map,1)
			File outputstatscoarse = new File("output/crossvalidation_outputs/statscoarse"+cvSet+".txt")
			println stats.getAccuracy()
			
			println "confusion matrix"+ stats.getConfusionMatrix()
			println "f1 for coarse " + stats.getF1()
			outputstatscoarse.append(" removing finetopic15,16,17,18->coarse(1)\n")
			outputstatscoarse.append("accuracy : "+stats.getAccuracy()+"\n")
			outputstatscoarse.append("overall f1 : "+ stats.getF1()+"\n")
			outputstatscoarse.append(  "confusion matrix"+ stats.getConfusionMatrix()+"\n")
			outputstatscoarse.append( "f1 for coarse 0 " + stats.getF1(0)+"\n")
			outputstatscoarse.append( "precision for coarse 0 " + stats.getPrecision(0)+"\n")
			outputstatscoarse.append( "recall for coarse 0 " + stats.getRecall(0)+"\n")
			
			outputstatscoarse.append( "f1 for coarse 1 " + stats.getF1(1)+"\n")
			outputstatscoarse.append( "precision for coarse 1 " + stats.getPrecision(1)+"\n")
			outputstatscoarse.append( "recall for coarse 1 " + stats.getRecall(1)+"\n")
			
			outputstatscoarse.append( "f1 for coarse 2 " + stats.getF1(2)+"\n")
			outputstatscoarse.append( "precision for coarse 2 " + stats.getPrecision(2)+"\n")
			outputstatscoarse.append( "recall for coarse 2 " + stats.getRecall(2)+"\n")
			
			outputstatscoarse.append( "f1 for coarse 3 " + stats.getF1(3)+"\n")
			outputstatscoarse.append( "precision for coarse 3" + stats.getPrecision(3)+"\n")
			outputstatscoarse.append( "recall for coarse 3" + stats.getRecall(3)+"\n")
			
			outputstatscoarse.append( "f1 for coarse 4 " + stats.getF1(4)+"\n")
			outputstatscoarse.append( "precision for coarse 4" + stats.getPrecision(4)+"\n")
			outputstatscoarse.append( "recall for coarse 4" + stats.getRecall(4)+"\n")

			outputstatscoarse.append( "f1 for coarse 5 " + stats.getF1(5)+"\n")
			outputstatscoarse.append( "precision for coarse 5" + stats.getPrecision(5)+"\n")
			outputstatscoarse.append( "recall for coarse 5" + stats.getRecall(5)+"\n")


			allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(finetopic_1))
			for(int k = 0;k<allGroundings.size();++k){
				GroundTerm [] grounding = allGroundings.get(k)
				System.out.println(grounding);
			}
		   map = new HashMap<GroundTerm, Integer>()
		   for(int k = 0;k<allGroundings.size();++k){
			   GroundTerm [] grounding = allGroundings.get(k)
			   String index = grounding[1].toString()
			   if(!map.containsKey(grounding[1])){
				   map.put(grounding[1], Integer.parseInt(index))}
		   }
			
						
			
			comparator.setResultFilter(new MaxValueFilter(finetopic_1, 1))
			//comparator.setThreshold(Double.MIN_VALUE)
			stats = comparator.compare(finetopic_1, map,1)
			File outputstats = new File("output/crossvalidation_outputs/statsfine"+cvSet+".txt")
			println stats.getAccuracy()

			println "confusion matrix"+ stats.getConfusionMatrix()
			println "f1 for fientopic " + stats.getF1()
			outputstats.append(" removing finetopic15,16,17,18->coarse(1)\n")
			outputstats.append("accuracy : "+stats.getAccuracy()+"\n")
			outputstats.append("overall f1 : "+ stats.getF1()+"\n")
			outputstats.append(  "confusion matrix"+ stats.getConfusionMatrix()+"\n")
			outputstats.append( "f1 for fientopic 0 " + stats.getF1(0)+"\n")
			outputstats.append( "precision for fientopic 0 " + stats.getPrecision(0)+"\n")
			outputstats.append( "recall for fientopic 0 " + stats.getRecall(0)+"\n")
			
			outputstats.append( "f1 for fientopic 1 " + stats.getF1(1)+"\n")
			outputstats.append( "precision for fientopic 1 " + stats.getPrecision(1)+"\n")
			outputstats.append( "recall for fientopic 1 " + stats.getRecall(1)+"\n")

			outputstats.append( "f1 for fientopic (2) " + stats.getF1(2)+"\n")
			outputstats.append( "precision for fientopic (2) " + stats.getPrecision(2)+"\n")
			outputstats.append( "recall for fientopic (2) " + stats.getRecall(2)+"\n")
			
			outputstats.append( "f1 for fientopic (3) " + stats.getF1(3)+"\n")
			outputstats.append( "precision for fientopic (3) " + stats.getPrecision(3)+"\n")
			outputstats.append( "recall for fientopic (3) " + stats.getRecall(3)+"\n")
			
			outputstats.append( "f1 for finetopic (4) " + stats.getF1(4)+"\n")
			outputstats.append( "precision for finetopic (4) " + stats.getPrecision(4)+"\n")
			outputstats.append( "recall for finetopic (4) " + stats.getRecall(4)+"\n")
			
			outputstats.append( "f1 for finetopic (5) " + stats.getF1(5)+"\n")
			outputstats.append( "precision for finetopic (5) " + stats.getPrecision(5)+"\n")
			outputstats.append( "recall for finetopic (5) " + stats.getRecall(5)+"\n")
			
			outputstats.append( "f1 for finetopic (6) " + stats.getF1(6)+"\n")
			outputstats.append( "precision for finetopic (6) " + stats.getPrecision(6)+"\n")
			outputstats.append( "recall for finetopic (6) " + stats.getRecall(6)+"\n")
			
			outputstats.append( "f1 for finetopic (7) " + stats.getF1(7)+"\n")
			outputstats.append( "precision for finetopic (7) " + stats.getPrecision(7)+"\n")
			outputstats.append( "recall for finetopic (7) " + stats.getRecall(7)+"\n")
			
			outputstats.append( "f1 for finetopic (8) " + stats.getF1(8)+"\n")
			outputstats.append( "precision for finetopic (8) " + stats.getPrecision(8)+"\n")
			outputstats.append( "recall for finetopic (8) " + stats.getRecall(8)+"\n")
			
			outputstats.append( "f1 for finetopic (9) " + stats.getF1(9)+"\n")
			outputstats.append( "precision for finetopic (9) " + stats.getPrecision(9)+"\n")
			outputstats.append( "recall for finetopic (9) " + stats.getRecall(9)+"\n")
			
			outputstats.append( "f1 for finetopic (10) " + stats.getF1(10)+"\n")
			outputstats.append( "precision for finetopic (10) " + stats.getPrecision(10)+"\n")
			outputstats.append( "recall for finetopic (10) " + stats.getRecall(10)+"\n")
			
			outputstats.append( "f1 for finetopic (11) " + stats.getF1(11)+"\n")
			outputstats.append( "precision for finetopic (11) " + stats.getPrecision(11)+"\n")
			outputstats.append( "recall for finetopic (11) " + stats.getRecall(11)+"\n")
			
			outputstats.append( "f1 for finetopic (12) " + stats.getF1(12)+"\n")
			outputstats.append( "precision for finetopic (12) " + stats.getPrecision(12)+"\n")
			outputstats.append( "recall for finetopic (12) " + stats.getRecall(12)+"\n")
			
			outputstats.append( "f1 for finetopic (13) " + stats.getF1(13)+"\n")
			outputstats.append( "precision for finetopic (13) " + stats.getPrecision(13)+"\n")
			outputstats.append( "recall for finetopic (13) " + stats.getRecall(13)+"\n")
			
			outputstats.append( "f1 for finetopic (14) " + stats.getF1(14)+"\n")
			outputstats.append( "precision for finetopic (14) " + stats.getPrecision(14)+"\n")
			outputstats.append( "recall for finetopic (14) " + stats.getRecall(14)+"\n")



			trainLabelsDB.close()
			trainDB.close()
			
		 
	}
	}


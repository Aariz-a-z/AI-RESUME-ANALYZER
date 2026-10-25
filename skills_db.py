"""Skill taxonomy: canonical name -> category, aliases, optional learning resource.

Aliases are matched case-insensitively against normalised text that preserves
``+ # . / -``, so entries like "c++", ".net" and "ci/cd" survive matching.

Deliberately omitted: bare "r" and bare "go". Both are common English words and
match far too often to be useful signal; the longer aliases carry them instead.
"""

# (canonical, category, aliases, learn_url)
SKILL_DB = [
    # ---- Programming Languages ----
    ("Python", "Programming Languages", ["python", "python3", "py3"], "https://docs.python.org/3/tutorial/"),
    ("Java", "Programming Languages", ["java", "java8", "java 11", "java 17"], "https://dev.java/learn/"),
    # Bare "js" is omitted on purpose: it fires inside node.js, next.js, vue.js
    # and three.js, inferring the language from every framework that embeds it.
    ("JavaScript", "Programming Languages", ["javascript", "es6", "ecmascript"], "https://javascript.info/"),
    ("TypeScript", "Programming Languages", ["typescript", "ts"], "https://www.typescriptlang.org/docs/handbook/intro.html"),
    ("C++", "Programming Languages", ["c++", "cpp"], "https://www.learncpp.com/"),
    ("C#", "Programming Languages", ["c#", "csharp", "c sharp"], "https://learn.microsoft.com/en-us/dotnet/csharp/"),
    ("C", "Programming Languages", ["c language", "c programming"], None),
    ("Go", "Programming Languages", ["golang", "go lang"], "https://go.dev/learn/"),
    ("Rust", "Programming Languages", ["rust", "rustlang"], "https://doc.rust-lang.org/book/"),
    ("Ruby", "Programming Languages", ["ruby"], None),
    ("PHP", "Programming Languages", ["php"], None),
    ("Swift", "Programming Languages", ["swift", "swiftui"], None),
    ("Kotlin", "Programming Languages", ["kotlin"], None),
    ("Scala", "Programming Languages", ["scala"], None),
    ("R", "Programming Languages", ["r programming", "rstudio", "cran", "r language"], None),
    ("MATLAB", "Programming Languages", ["matlab"], None),
    ("Shell Scripting", "Programming Languages", ["bash", "shell scripting", "shell script", "zsh", "powershell"], None),

    # ---- Machine Learning & AI ----
    ("Machine Learning", "Machine Learning & AI", ["machine learning", "ml", "predictive modeling", "predictive modelling", "supervised learning", "unsupervised learning"], "https://www.coursera.org/learn/machine-learning"),
    ("Deep Learning", "Machine Learning & AI", ["deep learning", "neural network", "neural networks", "cnn", "rnn", "lstm", "transformer architecture"], "https://www.deeplearning.ai/courses/deep-learning-specialization/"),
    ("Natural Language Processing", "Machine Learning & AI", ["natural language processing", "nlp", "text mining", "named entity recognition", "sentiment analysis"], "https://huggingface.co/learn/nlp-course"),
    ("Computer Vision", "Machine Learning & AI", ["computer vision", "image recognition", "object detection", "image classification"], None),
    ("Artificial Intelligence", "Machine Learning & AI", ["artificial intelligence", "ai"], None),
    ("Large Language Models", "Machine Learning & AI", ["large language model", "large language models", "llm", "llms", "gpt", "generative ai", "genai"], "https://huggingface.co/learn/nlp-course"),
    ("Prompt Engineering", "Machine Learning & AI", ["prompt engineering", "prompting"], None),
    ("RAG", "Machine Learning & AI", ["retrieval augmented generation", "rag pipeline", "vector database", "embeddings search", "semantic search"], None),
    ("TensorFlow", "Machine Learning & AI", ["tensorflow", "tf2"], "https://www.tensorflow.org/tutorials"),
    ("PyTorch", "Machine Learning & AI", ["pytorch", "torch"], "https://pytorch.org/tutorials/"),
    ("Keras", "Machine Learning & AI", ["keras"], None),
    ("scikit-learn", "Machine Learning & AI", ["scikit-learn", "scikit learn", "sklearn"], "https://scikit-learn.org/stable/getting_started.html"),
    ("Hugging Face", "Machine Learning & AI", ["hugging face", "huggingface", "transformers library"], "https://huggingface.co/learn/nlp-course"),
    ("XGBoost", "Machine Learning & AI", ["xgboost", "lightgbm", "catboost", "gradient boosting"], None),
    ("OpenCV", "Machine Learning & AI", ["opencv"], None),
    ("spaCy", "Machine Learning & AI", ["spacy"], None),
    ("NLTK", "Machine Learning & AI", ["nltk"], None),
    ("Reinforcement Learning", "Machine Learning & AI", ["reinforcement learning", "rlhf"], None),
    ("Time Series Analysis", "Machine Learning & AI", ["time series", "forecasting", "arima"], None),
    ("Recommender Systems", "Machine Learning & AI", ["recommender system", "recommendation engine", "collaborative filtering"], None),
    ("MLOps", "Machine Learning & AI", ["mlops", "model deployment", "model monitoring", "mlflow", "model serving"], None),
    ("Feature Engineering", "Machine Learning & AI", ["feature engineering", "feature selection", "feature extraction"], None),

    # ---- Data & Analytics ----
    ("Data Analysis", "Data & Analytics", ["data analysis", "data analytics", "exploratory data analysis", "eda"], None),
    ("Data Visualization", "Data & Analytics", ["data visualization", "data visualisation", "dashboarding", "dashboards"], None),
    ("Statistics", "Data & Analytics", ["statistics", "statistical analysis", "statistical modeling", "hypothesis testing", "regression analysis"], None),
    ("A/B Testing", "Data & Analytics", ["a/b testing", "ab testing", "experimentation", "split testing"], None),
    ("Pandas", "Data & Analytics", ["pandas"], None),
    ("NumPy", "Data & Analytics", ["numpy"], None),
    ("Excel", "Data & Analytics", ["excel", "microsoft excel", "spreadsheets", "pivot tables"], None),
    ("Tableau", "Data & Analytics", ["tableau"], None),
    ("Power BI", "Data & Analytics", ["power bi", "powerbi"], None),
    ("Looker", "Data & Analytics", ["looker", "looker studio"], None),
    ("Matplotlib", "Data & Analytics", ["matplotlib", "seaborn"], None),
    ("Plotly", "Data & Analytics", ["plotly", "dash"], None),
    ("Jupyter", "Data & Analytics", ["jupyter", "jupyter notebook", "google colab", "colab"], None),

    # ---- Databases ----
    ("SQL", "Databases", ["sql", "queries", "query optimization"], "https://www.postgresql.org/docs/current/tutorial.html"),
    ("PostgreSQL", "Databases", ["postgresql", "postgres", "psql"], "https://www.postgresql.org/docs/current/tutorial.html"),
    ("MySQL", "Databases", ["mysql", "mariadb"], None),
    ("MongoDB", "Databases", ["mongodb", "mongo", "nosql"], None),
    ("Redis", "Databases", ["redis", "memcached"], None),
    ("Elasticsearch", "Databases", ["elasticsearch", "opensearch", "elk stack"], None),
    ("Cassandra", "Databases", ["cassandra"], None),
    ("Oracle", "Databases", ["oracle database", "pl/sql"], None),
    ("SQL Server", "Databases", ["sql server", "mssql", "t-sql"], None),
    ("DynamoDB", "Databases", ["dynamodb"], None),
    ("Snowflake", "Databases", ["snowflake"], None),
    ("BigQuery", "Databases", ["bigquery", "big query"], None),
    ("Redshift", "Databases", ["redshift"], None),
    ("Neo4j", "Databases", ["neo4j", "graph database"], None),
    ("Database Design", "Databases", ["database design", "schema design", "data modeling", "data modelling", "normalization", "indexing"], None),

    # ---- Cloud & DevOps ----
    ("Amazon Web Services", "Cloud & DevOps", ["aws", "amazon web services", "ec2", "s3 bucket", "cloudformation"], "https://aws.amazon.com/training/"),
    ("Microsoft Azure", "Cloud & DevOps", ["azure", "microsoft azure"], "https://learn.microsoft.com/en-us/training/azure/"),
    ("Google Cloud", "Cloud & DevOps", ["google cloud", "gcp", "google cloud platform"], "https://cloud.google.com/learn"),
    ("Docker", "Cloud & DevOps", ["docker", "containerization", "containers", "containerisation"], "https://docs.docker.com/get-started/"),
    ("Kubernetes", "Cloud & DevOps", ["kubernetes", "k8s", "eks", "helm"], "https://kubernetes.io/docs/tutorials/"),
    ("Terraform", "Cloud & DevOps", ["terraform", "infrastructure as code", "iac"], "https://developer.hashicorp.com/terraform/tutorials"),
    ("CI/CD", "Cloud & DevOps", ["ci/cd", "cicd", "continuous integration", "continuous deployment", "continuous delivery"], None),
    ("Jenkins", "Cloud & DevOps", ["jenkins"], None),
    ("GitHub Actions", "Cloud & DevOps", ["github actions", "gitlab ci", "circleci"], None),
    ("Ansible", "Cloud & DevOps", ["ansible", "puppet", "chef config"], None),
    ("Linux", "Cloud & DevOps", ["linux", "unix", "ubuntu", "centos"], None),
    ("Nginx", "Cloud & DevOps", ["nginx", "apache server", "load balancing"], None),
    ("Monitoring", "Cloud & DevOps", ["prometheus", "grafana", "datadog", "observability", "new relic"], None),
    ("Serverless", "Cloud & DevOps", ["serverless", "aws lambda", "lambda functions", "cloud functions"], None),
    ("Cloud Architecture", "Cloud & DevOps", ["cloud architecture", "cloud infrastructure", "cloud migration", "cloud native"], None),

    # ---- Data Engineering ----
    ("ETL", "Data Engineering", ["etl", "elt", "data pipeline", "data pipelines", "ingestion"], None),
    ("Apache Spark", "Data Engineering", ["spark", "pyspark", "apache spark"], "https://spark.apache.org/docs/latest/quick-start.html"),
    ("Hadoop", "Data Engineering", ["hadoop", "hdfs", "mapreduce", "hive"], None),
    ("Kafka", "Data Engineering", ["kafka", "event streaming", "pub/sub"], "https://kafka.apache.org/quickstart"),
    ("Airflow", "Data Engineering", ["airflow", "dag orchestration", "prefect", "dagster"], "https://airflow.apache.org/docs/"),
    ("dbt", "Data Engineering", ["dbt", "data build tool"], None),
    ("Data Warehousing", "Data Engineering", ["data warehouse", "data warehousing", "data lake", "lakehouse", "star schema"], None),
    ("Databricks", "Data Engineering", ["databricks"], None),

    # ---- Web & Frontend ----
    ("React", "Web & Frontend", ["react", "react.js", "reactjs"], "https://react.dev/learn"),
    ("Angular", "Web & Frontend", ["angular", "angularjs"], None),
    ("Vue", "Web & Frontend", ["vue", "vue.js", "vuejs"], None),
    ("HTML", "Web & Frontend", ["html", "html5"], None),
    ("CSS", "Web & Frontend", ["css", "css3", "sass", "scss"], None),
    ("Next.js", "Web & Frontend", ["next.js", "nextjs"], None),
    ("Redux", "Web & Frontend", ["redux", "state management"], None),
    ("Tailwind CSS", "Web & Frontend", ["tailwind", "tailwind css"], None),
    ("Bootstrap", "Web & Frontend", ["bootstrap"], None),
    ("Responsive Design", "Web & Frontend", ["responsive design", "mobile-first", "cross-browser"], None),
    ("Streamlit", "Web & Frontend", ["streamlit", "gradio"], None),

    # ---- Backend & APIs ----
    ("REST APIs", "Backend & APIs", ["rest api", "rest apis", "restful", "api development", "api design"], None),
    ("GraphQL", "Backend & APIs", ["graphql", "apollo"], None),
    ("Node.js", "Backend & APIs", ["node.js", "nodejs", "node js"], "https://nodejs.org/en/learn"),
    ("Django", "Backend & APIs", ["django"], "https://docs.djangoproject.com/en/stable/intro/tutorial01/"),
    ("Flask", "Backend & APIs", ["flask"], None),
    ("FastAPI", "Backend & APIs", ["fastapi", "fast api"], "https://fastapi.tiangolo.com/tutorial/"),
    ("Spring Boot", "Backend & APIs", ["spring boot", "spring framework", "springboot"], None),
    ("Express", "Backend & APIs", ["express.js", "expressjs", "express framework"], None),
    ("Microservices", "Backend & APIs", ["microservice", "microservices", "service oriented"], None),
    (".NET", "Backend & APIs", [".net", "dotnet", "asp.net"], None),
    ("Ruby on Rails", "Backend & APIs", ["ruby on rails", "rails"], None),
    ("gRPC", "Backend & APIs", ["grpc", "protobuf", "protocol buffers"], None),
    ("WebSockets", "Backend & APIs", ["websocket", "websockets", "real-time communication"], None),

    # ---- Core CS ----
    ("Data Structures and Algorithms", "Core CS", ["dsa", "data structure", "data structures", "algorithm", "algorithms", "algorithmic"], "https://www.coursera.org/specializations/data-structures-algorithms"),
    ("Object-Oriented Programming", "Core CS", ["oop", "object oriented", "object-oriented"], None),
    ("System Design", "Core CS", ["system design", "architecture design", "scalable systems", "high availability"], None),
    ("Distributed Systems", "Core CS", ["distributed system", "distributed systems", "distributed computing", "consensus"], None),
    ("Operating Systems", "Core CS", ["operating system", "operating systems", "process scheduling", "memory management"], None),
    ("Computer Networks", "Core CS", ["computer network", "computer networks", "tcp/ip", "networking protocols", "http protocol"], None),
    ("Design Patterns", "Core CS", ["design pattern", "design patterns", "solid principles"], None),
    ("Concurrency", "Core CS", ["concurrency", "multithreading", "parallel processing", "async programming"], None),

    # ---- Tools & Practices ----
    ("Git", "Tools & Practices", ["git", "github", "gitlab", "bitbucket", "version control"], "https://git-scm.com/doc"),
    ("Agile", "Tools & Practices", ["agile", "scrum", "kanban", "sprint planning"], None),
    ("Jira", "Tools & Practices", ["jira", "confluence", "asana"], None),
    ("Unit Testing", "Tools & Practices", ["unit testing", "unit test", "pytest", "junit", "jest", "test coverage"], None),
    ("Test-Driven Development", "Tools & Practices", ["tdd", "test driven development", "test-driven"], None),
    ("Selenium", "Tools & Practices", ["selenium", "playwright", "cypress", "e2e testing"], None),
    ("Code Review", "Tools & Practices", ["code review", "peer review", "pull request"], None),
    ("Debugging", "Tools & Practices", ["debugging", "troubleshooting", "root cause analysis"], None),

    # ---- Professional ----
    ("Communication", "Professional", ["communication skills", "written communication", "verbal communication", "presentation skills"], None),
    ("Leadership", "Professional", ["leadership", "team lead", "led a team", "managing engineers"], None),
    ("Mentoring", "Professional", ["mentoring", "mentorship", "coaching", "onboarding juniors"], None),
    ("Collaboration", "Professional", ["collaboration", "cross-functional", "teamwork", "stakeholder"], None),
    ("Problem Solving", "Professional", ["problem solving", "analytical thinking", "critical thinking"], None),
    ("Project Management", "Professional", ["project management", "roadmap", "prioritization", "delivery management"], None),
]

CATEGORIES = []
for _name, _cat, _a, _u in SKILL_DB:
    if _cat not in CATEGORIES:
        CATEGORIES.append(_cat)

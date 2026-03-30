# Summary
Research-driven Data Engineer with experience across academic and industry settings. Skilled in designing and building scalable datapipelines, implementing data quality frameworks, and optimizing ELT processes. Proficient in Python, Spark, SQL, and AWS, with hands-on experience in large-scale data environments. Collaborates effectively across teams to enable efficient data access, analysis, anddecision-making. Fluent in English (C1) and certified B1 in German; strong communicator and team player.

# Skills
Languages: Python, TypeScript, SQL, Java, C, C++, R, PySpark
Frameworks: FastAPI, Streamlit
ML/AI: scikit-learn, PyTorch (basics), RAG (Retrieval-Augmented Generation), FAISS, embeddings, Hugging Face (LLM), Vector Databases, Data Parsing & Extraction, sentence-transformers, nbformat, BeautifulSoup
Cloud/DevOps: Docker, CI/CD (GitHub Actions, GitLab CI), Linux, AWS (certified), Azure DevOps, REST API, Terraform, GitPython
Project Management: JIRA, Trello, Agile Methodologies
IDEs & Code Editors: Visual Studio Code (VS Code), Jupyter Notebook
Data Visualization & Business Intelligence: MS PowerBI
Deployment / Web App Skills: Streamlit apps deployment, Hugging Face Spaces hosting, end-to-end AI pipeline management (ingestion → embedding → chatbot response)

# Experience
- Associate Consultant Data Engineer,Arcondis GmbH (Jun 2025 – Aug 2025)
Engineered an automated data pipeline to compute 30+ KPIs, integrating data from Monday.com via REST API and internal Abacus database using JDBC.
Built a helper automation to scan and isolate relevant tables within large-scale Abacus database—optimized execution time to ~3–4 minutes.
Fully automated the workflow end-to-end, enabling scheduled KPI updates with zero manual intervention and improved reporting cadence.
Interacted with stakeholders to understand the requirements and to give weekly updates.

- Online Tutor,Self Employed (Jan 2024 – May 2025)
Provide expert tutoring and academic supervision to bachelor's and master's students in Data Engineering, Cloud Computing, DevOps, and Databases.
Design and deliver structured lesson plans and presentations to facilitate comprehensive learning experiences.
Conduct collaborative sessions, including pair programming, to enhance student engagement and foster an interactive learning environment.

- Data Engineer,Roche Diagnostics GmbH (Feb 2023 – Jul 2023 Penzberg, Germany)
Built a Python ETL pipeline for clinical pathological datasets (OCR, PDF, text) to detect data drift.
Applied NLP and text mining techniques for transformation of unstructured to structured data.
Utilized ML algorithms, with Word2Vec enhancing results by 99.5%.
Implemented Cosine similarity vectors to analyze and quantify differences across reports.
Automated deployment processes using DevOps tools (Git, AWS, Docker), optimizing efficiency and scalability.

- Apprentice Data Engineer,Roche Diagnostics GmbH (Jun 2022 – Aug 2022 Penzberg, Germany)
Collaborated in Agile development through daily stand-ups and sprint planning.
Established a cloud-based ETL data pipeline using AWS Glue, S3, and Athena.
Utilized SQL optimization techniques in Athena for ad hoc data analysis.
Used PySpark for ETL into a centralized data lake.
Boosted uptime from 48% to 87% using Amazon CloudWatch monitoring.

- Software Engineer,Fortress6 Technologies (Jun 2019 – May 2021, India)
Collaborated with project managers, engineers, and stakeholders to support ongoing projects.
Reduced data retrieval time by 60% through SQL optimization
Automated AWS microservice deployment using Terraform (IaC) for 13+ ISPs.
Implemented CI/CD pipelines using Git for 60+ clients.
Mentored 5 junior developers, improving code quality by 25%.
Enhanced backend development, positively impacting company performance.

# Education
- MSc. Data Engineering, Jacobs (Constructor) University, Bremen, Germany
- B.Tech Computer Science & Engineering, Uttarakhand Technical University, India

# LANGUAGE
- English: Fluent
- Hindi: Native
- German: B1 certified, B2 course attended

# AWARDS/ CERTIFICATIONS
- Recipient of 100% scholarship for MSc: Roche Cooperative Study Program
- Academic merit scholarship: Jacobs University Bremen
- AWS Cloud Computing and Deployment: WebTek Labs Pvt. Ltd.
- A+ in MySQL training from Microsoft, ranking among the top 5%: Microsoft Technology Associate


## Roche (Working Student / Data Engineer)

### Project 1: Guideline Digitalization — JSON Backend Pipeline

**Objective:** Convert oncology and CCN reports from PDF format into structured, queryable data.

**Pipeline Overview:**
- Used Roche's internal Harvester tool to extract content from PDF-based oncology and CCN reports and convert them into JSON format.
- Ingested JSON files into **AWS S3** bucket as the data lake entry point.
- Built an **AWS Glue ETL pipeline** using **PySpark** to transform JSON files into **Parquet format** for optimized storage and query performance.
- Stored Parquet output back into S3.
- Used **AWS Glue Crawler** to automatically create and catalog tables from the Parquet files, eliminating the need for manual schema definition.
- Connected **Amazon Athena** to the Glue Data Catalog, selected the relevant database, and wrote optimized SQL queries to extract structured information.
- Extracted nodes, sub-nodes, headings, and clinical data clusters required for data curation and analysis.

**Technologies:** AWS S3, AWS Glue, AWS Athena, PySpark, JSON, Parquet

---

### Project 2: Detecting Data Drift in Clinical NLP Pipelines

**Objective:** Automatically detect when incoming pathological reports deviate from the distribution of surgical pathological reports used in training.

**Problem Statement:**
Clinical NLP pipelines are trained on surgical pathological reports. Over time, reports from other departments and sources enter the system. The goal was to detect when incoming data drifts away from the expected training distribution.

**Pipeline Overview:**
- Collected surgical pathological reports and other pathological reports from various departments via a data warehouse / data lake.
- Converted scanned OCR images of reports into text format.
- Ingested all data into a plain Python pipeline.
- Divided data into three sets:
  - **Training Data:** Surgical pathological reports only
  - **Test Data (In-distribution):** Reports with high similarity to training data
  - **Out-Test Data (Out-of-distribution):** Reports with low similarity to training data
- Split training data into **80% training / 20% validation**.
- Applied **Word2Vec** and **TF-IDF** to vectorize reports.
- Used **Cosine Similarity** to compute similarity scores between incoming reports and the training distribution.
- Established a **similarity threshold** based on internal variance within the training data.
- Compared incoming reports against this threshold to automatically classify whether data drift was detected or not.

**Technologies:** Python, Word2Vec, TF-IDF, Cosine Similarity, OCR, NLP

**GitHub:** https://github.com/tanzilalam23/Detecting-Data-Drift-_Automated

---

## Career Gap (approx. 1.5 Years) — Mentoring, Teaching & Leadership

During the career gap, Mohammad took on a voluntary mentoring and teaching role,
demonstrating strong leadership, communication, and people development skills.

### Teaching & Mentoring
- Independently designed and delivered structured learning programs for Master's
  students from Germany and India pursuing careers in Data Engineering, Data
  Science, Cloud Computing, and CI/CD pipelines.
- Mentored students on their **Master's thesis projects** at institutions
  including Fraunhofer, guiding them through technical challenges and
  research methodology.

### Leadership & Soft Skills Demonstrated
- **Mentorship Leadership:** Took full ownership of students' learning journeys
  without any institutional support or compensation — driven purely by passion
  for knowledge sharing.
- **Communication:** Translated complex technical concepts (AWS, pipelines,
  data science workflows) into beginner-friendly, structured lessons for
  diverse audiences across two countries.
- **Patience & Empathy:** Adapted teaching style to each student's background,
  pace, and learning goals.
- **Accountability:** Students consistently progressed and achieved their
  academic and career goals under Mohammad's guidance.
- **Cross-cultural Collaboration:** Worked with students from Germany and India,
  navigating different academic systems, languages, and expectations.
- **Initiative & Self-motivation:** Proactively identified a gap in technical
  education among aspiring data professionals and stepped in to fill it.

### Impact
- Successfully guided multiple students toward careers in Data Engineering
  and Data Science.
- Supported Master's thesis completions at top German research institutions.
- Built a reputation as a trusted technical mentor within the community.

---

## Arcondis (Senior Consultant Data Engineer) — June 2025 to August 2025

**Note:** Employment ended due to a company-wide layoff of 300 employees.

### Project: Automated ERP/CRM Data Dashboard System

**Objective:** Build an automated data pipeline and dashboard system pulling data from ERP and CRM systems.

**Key Contributions:**
- Integrated data from **Monday.com** (CRM) and **Abacus** (ERP) using **JDBC connection**.
- Built an **automated table search system** in Abacus that scanned all database tables and returned only relevant tables — reducing manual search effort significantly (execution time: ~2.5 to 3 minutes).
- Used **Python** to ingest and process data from these systems.
- Implemented data transfer using **SMTP** and **REST API** connections.
- Also contributed to an external **financial data project**, demonstrating flexibility outside core Data Engineering scope.
- Participated in **weekly presentations**, **biweekly sprint reviews**, and Agile ceremonies.

**Technologies:** Python, JDBC, REST API, SMTP, Monday.com, Abacus ERP, Agile/Scrum

---

## LinkedIn Recommendations

### Nino Mandela Bachmann — GxP Expert, Digital & AI Consultant at Arcondis (Direct Manager)
*"During the time I worked with Mohammad, I consistently felt his dedication and passion for Data Engineering. He contributed valuable insights to the team and played a key role in a critical project involving financial data. Additionally, he demonstrated great flexibility by stepping in to support a project completely unrelated to Data Engineering, showing his adaptability and team spirit."*

### Ole Eigenbrod — Senior Colleague at Roche
*"Mohammad has tackled a project regarding cloud-based data pipelines in the digital healthcare area. He has shown great technical expertise (data engineering and data analytics), strong dedication and a natural adoption of agile ways of working. His eagerness to learn and to provide solutions have paved the way for a successful outcome of the project work. Mohammad has been a great addition to the team!"*


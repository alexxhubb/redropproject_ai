import math
from collections import defaultdict

# ============================================================================
# SKILL_ALIASES - Use Exactly As Provided in Problem Sheet
# ============================================================================
SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    
    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",
    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",
    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics",
    "stats": "statistics",
    "regression": "regression",
    "clustering": "clustering",
    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",
    "pandas": "pandas",
    "numpy": "numpy",
    
    # Web — Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
    "redux": "redux",
    "tailwind": "tailwind",
    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",
    "jest": "jest",
    "graphql": "graphql",
    
    # Web — Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",
    "microservices": "microservices",
    
    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",
    "aws": "aws",
    
    # Mobile
    "android": "android",
    "firebase": "firebase",
    
    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",
    "data structure": "data_structures",
    "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    
    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# ============================================================================
# RESUME DATA - 10 Candidates
# ============================================================================
RESUMES = {
    "Arjun Sharma": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta": "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh": "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta": "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer": "python, R, statistics, ML, regression, clustering, Power-BI",
}

# ============================================================================
# JOB DESCRIPTIONS - 3 Jobs
# ============================================================================
JOB_DESCRIPTIONS = {
    "JD-1": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics",
    "JD-2": "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis",
    "JD-3": "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS",
}

# ============================================================================
# STEP 1: SKILL NORMALIZATION
# ============================================================================
def normalize_skills(raw_skills_string):
    """
    Split skills on commas, lowercase, apply alias mapping, discard unknown tokens.
    Multi-word phrases must be matched before single-token matches.
    """
    # Split by comma and strip whitespace
    tokens = [token.strip() for token in raw_skills_string.split(",")]
    
    normalized = []
    for token in tokens:
        # Convert to lowercase
        token_lower = token.lower()
        
        # Try to find in aliases (exact match)
        if token_lower in SKILL_ALIASES:
            canonical = SKILL_ALIASES[token_lower]
            normalized.append(canonical)
        else:
            # Token not in alias map - discard it
            pass
    
    return normalized

# ============================================================================
# STEP 2: DEDUPLICATION
# ============================================================================
def deduplicate_skills(normalized_skills):
    """
    Remove duplicate canonical skills from each resume.
    """
    return list(dict.fromkeys(normalized_skills))  # Preserves order, removes duplicates

# ============================================================================
# STEP 3: VOCABULARY CONSTRUCTION
# ============================================================================
def build_vocabulary(all_resumes_dict):
    """
    Create shared vocabulary from all normalized + deduplicated resume skills.
    Sort alphabetically.
    """
    all_skills = set()
    
    for candidate, raw_skills in all_resumes_dict.items():
        normalized = normalize_skills(raw_skills)
        deduplicated = deduplicate_skills(normalized)
        all_skills.update(deduplicated)
    
    # Sort alphabetically
    vocabulary = sorted(list(all_skills))
    return vocabulary

# ============================================================================
# STEP 4: TF-IDF COMPUTATION
# ============================================================================
def compute_tf_idf_vectors(all_resumes_dict, vocabulary):
    """
    Compute TF-IDF vectors for all resumes.
    TF = 1 / N (after deduplication, each skill appears once)
    IDF = ln(10 / df(skill))
    TF-IDF = TF × IDF
    """
    # First pass: normalize and deduplicate all resumes
    resume_normalized_skills = {}
    df = defaultdict(int)  # Document frequency
    
    for candidate, raw_skills in all_resumes_dict.items():
        normalized = normalize_skills(raw_skills)
        deduplicated = deduplicate_skills(normalized)
        resume_normalized_skills[candidate] = deduplicated
        
        # Count document frequency
        for skill in set(deduplicated):
            df[skill] += 1
    
    # Second pass: compute TF-IDF vectors
    tfidf_vectors = {}
    
    for candidate, skills in resume_normalized_skills.items():
        N = len(skills)  # Total unique skills in resume
        vector = [0.0] * len(vocabulary)
        
        for skill in skills:
            if skill in vocabulary:
                skill_idx = vocabulary.index(skill)
                
                # TF = 1 / N
                tf = 1.0 / N if N > 0 else 0
                
                # IDF = ln(10 / df)
                idf = math.log(10.0 / df[skill]) if df[skill] > 0 else 0
                
                # TF-IDF
                vector[skill_idx] = tf * idf
        
        tfidf_vectors[candidate] = vector
    
    return tfidf_vectors

# ============================================================================
# STEP 5: BUILD JD VECTORS (Binary)
# ============================================================================
def build_jd_vectors(jd_dict, vocabulary):
    """
    Create binary vectors for JDs over the same vocabulary.
    1 if skill is in JD, 0 otherwise.
    """
    jd_vectors = {}
    
    for jd_id, jd_skills_string in jd_dict.items():
        # Split JD skills on commas
        jd_skills = [skill.strip().lower() for skill in jd_skills_string.split(",")]
        
        # Map to canonical form using aliases
        canonical_jd_skills = []
        for skill in jd_skills:
            if skill in SKILL_ALIASES:
                canonical_jd_skills.append(SKILL_ALIASES[skill])
        
        # Create binary vector
        vector = [0] * len(vocabulary)
        for i, vocab_skill in enumerate(vocabulary):
            if vocab_skill in canonical_jd_skills:
                vector[i] = 1
        
        jd_vectors[jd_id] = vector
    
    return jd_vectors

# ============================================================================
# STEP 6: COSINE SIMILARITY & RANKING
# ============================================================================
def cosine_similarity(resume_vector, jd_vector):
    """
    Cosine(A, B) = (A · B) / (|A| × |B|)
    """
    # Dot product
    dot_product = sum(a * b for a, b in zip(resume_vector, jd_vector))
    
    # Euclidean norms
    magnitude_a = math.sqrt(sum(a * a for a in resume_vector))
    magnitude_b = math.sqrt(sum(b * b for b in jd_vector))
    
    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)

def rank_candidates(tfidf_vectors, jd_vectors):
    """
    Compute cosine similarity for all resumes against each JD.
    Return top 3 candidates per JD, sorted by score (descending) then by name (ascending).
    """
    results = {}
    
    for jd_id, jd_vector in jd_vectors.items():
        scores = []
        
        for candidate, resume_vector in tfidf_vectors.items():
            similarity = cosine_similarity(resume_vector, jd_vector)
            scores.append((candidate, similarity))
        
        # Sort by score descending, then by name ascending (for tie-breaking)
        scores.sort(key=lambda x: (-x[1], x[0]))
        
        # Get top 3
        top_3 = scores[:3]
        results[jd_id] = top_3
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*80)
    print("RESUME MATCHING ENGINE")
    print("="*80)
    
    # Step 1 & 2: Normalize and Deduplicate
    print("\n[STEP 1-2] NORMALIZING & DEDUPLICATING SKILLS")
    print("-"*80)
    normalized_resumes = {}
    for candidate, raw_skills in RESUMES.items():
        normalized = normalize_skills(raw_skills)
        deduplicated = deduplicate_skills(normalized)
        normalized_resumes[candidate] = deduplicated
        print(f"{candidate}: {deduplicated}")
    
    # Step 3: Build Vocabulary
    print("\n[STEP 3] BUILDING VOCABULARY")
    print("-"*80)
    vocabulary = build_vocabulary(RESUMES)
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Vocabulary: {vocabulary}")
    
    # Step 4: Compute TF-IDF
    print("\n[STEP 4] COMPUTING TF-IDF VECTORS")
    print("-"*80)
    tfidf_vectors = compute_tf_idf_vectors(RESUMES, vocabulary)
    for candidate, vector in tfidf_vectors.items():
        non_zero = [(vocabulary[i], vector[i]) for i in range(len(vocabulary)) if vector[i] > 0]
        print(f"{candidate}: {non_zero}")
    
    # Step 5: Build JD Vectors
    print("\n[STEP 5] BUILDING JD BINARY VECTORS")
    print("-"*80)
    jd_vectors = build_jd_vectors(JOB_DESCRIPTIONS, vocabulary)
    for jd_id, vector in jd_vectors.items():
        skills_in_jd = [vocabulary[i] for i in range(len(vocabulary)) if vector[i] == 1]
        print(f"{jd_id}: {skills_in_jd}")
    
    # Step 6: Compute Similarity & Rank
    print("\n[STEP 6] COMPUTING COSINE SIMILARITY & RANKING")
    print("-"*80)
    results = rank_candidates(tfidf_vectors, jd_vectors)
    
    # Print Final Results in Expected Format
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    for jd_id in ["JD-1", "JD-2", "JD-3"]:
        jd_name = {
            "JD-1": "Kakao (ML Engineer)",
            "JD-2": "Naver (Backend Engineer)",
            "JD-3": "Line (Frontend Engineer)"
        }[jd_id]
        
        print(f"\n{jd_id} — {jd_name}")
        top_3 = results[jd_id]
        result_str = ", ".join([f"{name}({score:.2f})" for name, score in top_3])
        print(result_str)

if __name__ == "__main__":
    main()

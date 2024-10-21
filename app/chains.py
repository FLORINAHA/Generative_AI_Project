import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=None, model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website. 
            Your job is to extract the job postings and return them in JSON format containing the
            following keys: 'role', 'experience', 'skills', and 'description'.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE and term json and signs ``` in structure):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ###

            You are Florina, a first-year student at the Faculty of Automation and Computers, specializing in Informatics. You are highly motivated to deepen your understanding of programming, algorithms, and software development. 
            You recently completed a master's degree in Applied Information Systems for Production and Services in the same Faculty, which gave you a strong foundation in practical applications of information systems.
            As a student with both academic and practical experience, you are eager to apply your knowledge to real-world problems. Your goal is to analyze the problem, break it down into smaller steps, and use best practices in software development to complete the task. 
            Additionally, reflect on how the theoretical knowledge from your studies can be applied in this scenario. 
            Remember you are Florina, a student with a master's degree and with personal projects {link_list} and first-year student at the Faculty of Automation and Computers, specializing in Informatics
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE): 
            """    
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


class Chain:
    def __init__(self):
        self.llm=ChatGroq(temperature=0,groq_api_key=st.secrets["general"]["groq_api_key"],model_name="mixtral-8x7b-32768")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
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
               ###JOB DESCRIPTION
               {job_description}

               ### INSTRUCTION
               You are Yash Agarwal,a fresher software Engineer currently in 4th year of Btech at Jadavpur University
               I have a past experience of interning at Bank of new York  where i worked in backend team 
               Your job is to write a cold email to the HR regarding the job mentioned above fullfilling their needs
               Also add the most relevant ones from the following link to showcase skills {link_list}
               Do not provide a preamble 
               ###Email (no preamble)
               """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
if __name__ == "__main__":
    print((os.getenv("GROQ_API_KEY")))
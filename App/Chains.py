import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("LLaMa_API_KEY"), model_name="llama-3.3-70b-versatile")

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
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Nayeem Hossen Jim, a passionate ML engineer and game developer. Your role is to craft a professional cold email to the client regarding the job mentioned above. In the email, highlight AtliQ’s expertise in meeting their business needs, while emphasizing the value we bring through our AI-driven solutions and software consulting services.

            The email should reflect AtliQ’s proven track record in empowering businesses by integrating advanced technologies, optimizing processes, and delivering customized automation solutions that ensure scalability, cost efficiency, and enhanced productivity. Make sure to showcase how our tailored solutions have helped other businesses achieve remarkable success.

            Additionally, you should incorporate relevant examples from the portfolio links provided below, emphasizing our experience and showcasing our successful projects in areas that directly align with the client’s requirements.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("LLaMa_API_KEY"))
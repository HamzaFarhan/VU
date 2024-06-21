import json
import random
import time
import typing as _t
from enum import Enum
from pathlib import Path

import anthropic
import instructor
import openai
from dotenv import load_dotenv
from vu_models import (
    MAX_CONCEPTS,
    MAX_SUBTOPICS,
    MIN_CONCEPTS,
    MIN_SUBTOPICS,
    BaseQuestion,
    CreatedTopic,
    Question,
    Subtopic,
    Topic,
    Topics,
)

random.seed(42)

load_dotenv()

ask_oai = instructor.from_openai(openai.OpenAI())
ask_cld = instructor.from_anthropic(anthropic.Anthropic())

SYSTEM_PREFIX = "You are a world class math course instructor."
MIN_WORDS = 3
MAX_WORDS = 5
QUESTIONS_PER_FOLDER = 80
ATTEMPTS = 2
MAX_TOKENS = 2048
TEMPERATURE = 0.3


class ModelName(str, Enum):
    GPT_3 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-5-sonnet-20240620"


def chat_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def system_message(content: str) -> dict[str, str]:
    return chat_message(role="system", content=content)


def user_message(content: str) -> dict[str, str]:
    return chat_message(role="user", content=content)


def assistant_message(content: str) -> dict[str, str]:
    return chat_message(role="assistant", content=content)


def merge_same_role_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return []
    new_messages = []
    last_message = None
    for message in messages:
        if last_message is None:
            last_message = message
        elif last_message["role"] == message["role"]:
            last_message["content"] += "\n\n" + message["content"]
        else:
            new_messages.append(last_message)
            last_message = message
    if last_message is not None:
        new_messages.append(last_message)
    return new_messages


def oai_response(response) -> str:
    try:
        return response.choices[0].message.content
    except Exception:
        return response


def claude_response(response) -> str:
    try:
        return response.content[0].text
    except Exception:
        return response


def ask_cld_or_oai(
    ask_cld: instructor.Instructor,
    ask_oai: instructor.Instructor,
    messages: list[dict[str, str]],
    system: str = "",
    model: ModelName = ModelName.GPT_4O,
    response_model: _t.Optional[type] = None,
    attempts: int = ATTEMPTS,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    validation_context: dict = {},
    ask_kwargs: dict = {},
):
    ask_kwargs["model"] = model
    ask_kwargs["max_retries"] = attempts
    ask_kwargs["max_tokens"] = max_tokens
    ask_kwargs["temperature"] = temperature
    ask_kwargs["response_model"] = response_model
    ask_kwargs["validation_context"] = validation_context
    # print(f"ASK_KWARGS:\n{ask_kwargs}")
    try:
        if "gpt" in ask_kwargs["model"].lower():
            if system:
                messages.insert(0, system_message(system))
            res = ask_oai.create(
                messages=messages,  # type: ignore
                **ask_kwargs,
            )
            if response_model is None:
                return oai_response(res)
            return res
        else:
            res = ask_cld.create(
                system=system,
                messages=merge_same_role_messages(messages),  # type: ignore
                **ask_kwargs,
            )
            if response_model is None:
                return claude_response(res)
            return res
    except Exception as e:
        print(f"Error in ask_cld_or_oai. Messages: {messages}")
        print(e)
        return None


def create_topics(
    outline: list[str] | str,
    model: ModelName = ModelName.GPT_4O,
    topics_file: str | Path = "",
) -> Topics:
    topic_system = f"""\
    {SYSTEM_PREFIX}
    You'll be given a topic description from a course outline and you have to generate a {MIN_WORDS}-{MAX_WORDS} word topic name that encapsulates the description.
    Then, generate {MIN_SUBTOPICS}-{MAX_SUBTOPICS} subtopics for the topic. Also {MIN_WORDS}-{MAX_WORDS} words each.
    Then for each subtopic, generate {MIN_CONCEPTS}-{MAX_CONCEPTS} concepts. Also {MIN_WORDS}-{MAX_WORDS} words each. The concepts should be related to the subtopic.
    Think of concepts as the smallest unit of knowledge that can be taught from the subtopic. And add a verb to the concept to make it actionable.
    For example:
    "Calculate Derivatives" instead of "Derivatives".
    "Identify Finite Sets" instead of "Finite Sets".
    "Find the y-intercept" instead of "y-intercept".
    The subtopics and concepts should be in the correct order.
    """
    topics = Topics()
    if isinstance(outline, str):
        outline = outline.split("\n")
    for line in outline:
        topic: CreatedTopic | None = ask_cld_or_oai(
            ask_cld=ask_cld,
            ask_oai=ask_oai,
            model=model,
            response_model=CreatedTopic,
            system=topic_system,
            messages=[
                user_message(
                    f"<topic_description>\n{line.strip()}\n</topic_description>"
                )
            ],
        )  # type: ignore
        if topic is None:
            continue
        topic2 = Topic(topic=topic.name)
        for subtopic in topic.subtopics:
            subtopic2 = Subtopic(topic=topic.name, subtopic=subtopic.name)
            subtopic2.add_concepts(subtopic.concepts)
            topic2.add_subtopics(subtopic2)
        topics.add_topics(topic2)
        if topics_file:
            with open(Path(topics_file).with_suffix(".json"), "w") as f:
                json.dump(topics.model_dump(), f, indent=2)
    return topics


def topics_json_to_human_json(topics_file: str | Path) -> Path:
    topics_file = Path(topics_file)
    human_file = topics_file.with_stem(topics_file.stem + "_human")
    topics = Topics(**(json.loads(topics_file.read_text())))
    topics_dict = {}
    for topic in topics.topics.values():
        topics_dict[topic.topic] = {}
        for subtopic in topic.subtopics.values():
            topics_dict[topic.topic][subtopic.subtopic] = [
                concept.concept for concept in subtopic.concepts.values()
            ]
    human_file.write_text(json.dumps(topics_dict, indent=2))
    return human_file


def human_json_to_topics_json(human_file: str | Path) -> Path:
    human_file = Path(human_file)
    topics_file = human_file.with_stem(human_file.stem.replace("_human", ""))
    human = json.loads(human_file.read_text())
    topics = Topics()
    for topic_name, subtopics in human.items():
        topic = Topic(topic=topic_name)
        for subtopic_name, concepts in subtopics.items():
            subtopic = Subtopic(topic=topic_name, subtopic=subtopic_name)
            subtopic.add_concepts(concepts=concepts)
            topic.add_subtopics(subtopics=subtopic)
        topics.add_topics(topics=topic)
    topics_file.write_text(json.dumps(topics.model_dump(), indent=2))
    return topics_file


def create_subquestions(question: Question, model: ModelName = ModelName.GPT_4O):
    system = """
    You are world class math instructor.
    You will be given a problem and its solution and your job is to break down the solution into steps and explain each step in detail.
    I will be using your steps as a transcript for text-to-speech. So, make sure the steps are in plain English and easy to understand.
    The text-to-speech model would get confused by back slashes and other symbols. So, make sure to remove them.
    For example, instead of writing $x^2$, write x squared. For fractions, write x over y. etc. No brackets or other symbols.
    For math operations, wrtie plus, minus, times, divided by, etc. instead of +, -, *, /, etc.
    If a number is -5, write negative 5. If a number is 2.5, write 2 point 5.
    But make sure to not lose the meaning of the math expression. A student should be able to understand the math expression from the text-to-speech.
    No more than 5 steps per problem.
    Make sure the final answer is included in the last step.
    """

    messages = [
        user_message(
            f"<problem>\n{question.problem}\n</problem>\n\n<solution>\n{question.solution}\n</solution>"
        )
    ]
    try:
        subquestions = ask_cld_or_oai(
            ask_cld=ask_cld,
            ask_oai=ask_oai,
            messages=messages,
            system=system,
            model=model,
            response_model=list[BaseQuestion],
        )
    except Exception as e:
        print(e)
        subquestions = []
    question.subquestions = subquestions  # type: ignore


def create_code(question: Question, model: ModelName = ModelName.GPT_4O):
    system = Path("code_prompt.txt").read_text()
    user = f"<problem>\n{question.problem}\n</problem>\n\n<solution>\n{question.solution}\n</solution>"
    if question.subquestions:
        user += "\n\n<subquestions>"
        for i, subquestion in enumerate(question.subquestions, start=1):
            user += f"\n<subquestion_{i}>\n{subquestion}\n</subquestion_{i}>"
        user += "\n</subquestions>"
    messages = [user_message(user), assistant_message("```python")]
    try:
        code = ask_cld_or_oai(
            ask_cld=ask_cld,
            ask_oai=ask_oai,
            messages=messages,
            system=system,
            model=model,
            response_model=str,
        )  # type: ignore
        question.code = code  # type: ignore
    except Exception as e:
        print(e)


def load_questions(
    questions_dir: str | Path,
    questions_per_folder: int = QUESTIONS_PER_FOLDER,
    folder_names: list[str] | None = None,
    shuffle: bool = True,
) -> list[BaseQuestion]:
    folder_names = folder_names or []
    questions_dir = Path(questions_dir)
    questions = []
    for f in questions_dir.iterdir():
        if f.is_dir():
            if folder_names and f.name not in folder_names:
                continue
            questions += list(f.glob("*.json"))[:questions_per_folder]
        else:
            questions.append(f)
    if shuffle:
        random.shuffle(questions)
    return [
        BaseQuestion(**json.loads(question_file.read_text()))
        for question_file in questions
    ]


def assign_questions(
    questions_dir: str | Path,
    topics: Topics | None = None,
    topics_file: str | Path = "",
    assigner_model: ModelName = ModelName.HAIKU,
    subquestion_model: ModelName = ModelName.GPT_4O,
    code_model: ModelName = ModelName.GPT_4O,
    assigned_questions_file: str | Path = "assigned_questions.json",
    questions_per_folder: int = QUESTIONS_PER_FOLDER,
    folder_names: list[str] | None = None,
    num_questions: int | None = None,
) -> Topics:
    """
    Assigns questions to topics based on their content and updates the topics and assigned questions.

    Parameters:
    questions_dir (str | Path): The directory where questions are stored.
    topics (Topics | None): The current set of topics, if already loaded.
    topics_file (str | Path): The file path to load or save topics.
    assigner_model (ModelName): The model used for assigning questions to topics.
    subquestion_model (ModelName): The model used for generating subquestions.
    code_model (ModelName): The model used for generating cthe python code to make random questions.
    assigned_questions_file (str | Path): The file path to save assigned questions.
    questions_per_folder (int): The number of questions per folder to process.
    folder_names (list[str] | None): The list of folder names to process. If None, processes all folders.
    num_questions (int | None): The total number of questions to assign. If None, assigns all questions.

    Returns:
    Topics: The updated topics with newly assigned questions.

    Raises:
    AssertionError: If neither topics nor topics_file is provided, or if questions_dir does not exist.

    This function processes questions from a specified directory, assigns them to topics based on the assigner model,
    generates subquestions and code snippets using specified models, and updates the topics and assigned questions files accordingly.
    """
    assert Path(questions_dir).exists(), f"{questions_dir} does not exist"
    assert topics or topics_file, "Either topics or topics_file must be provided."
    if topics_file:
        topics_file = Path(topics_file).with_suffix(".json")
        if topics_file.exists():
            print(f"Loading topics from {topics_file}")
            topics = Topics(**json.loads(topics_file.read_text()))
    assert topics, "Topics must be provided."
    questions = load_questions(
        questions_dir=questions_dir,
        questions_per_folder=questions_per_folder,
        folder_names=folder_names,
    )
    assigned_questions_file = Path(assigned_questions_file).with_suffix(".json")
    assigned_questions = (
        json.loads(assigned_questions_file.read_text())
        if assigned_questions_file.exists()
        else {}
    )
    assigned_questions["used"] = assigned_questions.get("used", 0)
    assigned_questions["questions"] = [
        Question(**question) for question in assigned_questions.get("questions", [])
    ]

    questions_system = f"""\
    {SYSTEM_PREFIX}
    You'll be given the problem and solution of a question and list of topic_subtopic_concept objects.
    Based on your knowledge of math, you have to decide which topic_subtopic_concept the question belongs to.
    """
    num_questions = num_questions or len(questions)
    used_index = assigned_questions["used"]
    for i, question in enumerate(
        questions[used_index:num_questions], start=used_index + 1
    ):
        print(f"Question {i}")
        concept_strs = [
            str(concept)
            for concept in topics.concepts.values()
            if len(concept.questions) < 3
        ]
        if not concept_strs:
            break

        objects_str = "\n\n".join(concept_strs)
        concept_ids = list(topics.concepts.keys())
        messages = [
            user_message(
                f"<question>\n{question}\n</question>\n\n<objects>\n{objects_str}\n</objects>"
            )
        ]
        try:
            belongs_to: str = ask_cld_or_oai(
                ask_cld=ask_cld,
                ask_oai=ask_oai,
                messages=messages,
                system=questions_system,
                model=assigner_model,
                response_model=_t.Literal[*concept_ids],  # type: ignore
            )
            split = belongs_to.split("_")
            assigned_question = Question(
                topic=split[0],
                subtopic=split[1],
                concept=split[2],
                problem=question.problem,
                solution=question.solution,
            )
            create_subquestions(question=assigned_question, model=subquestion_model)
            create_code(question=assigned_question, model=code_model)
            assigned_questions["questions"].append(assigned_question.model_dump())
            topics.add_questions(assigned_question)
            if topics_file:
                with open(Path(topics_file).with_suffix(".json"), "w") as f:
                    json.dump(topics.model_dump(), f, indent=2)
            with open(assigned_questions_file, "w") as f:
                json.dump(
                    {"used": i, "questions": assigned_questions["questions"]},
                    f,
                    indent=2,
                )
            time.sleep(0.3)
        except Exception as e:
            print(e)
            continue

    return topics


def add_dependencies(
    topics: Topics | None = None,
    topics_file: str | Path = "",
    model: ModelName = ModelName.HAIKU,
    prerequisities_file: str | Path = "prerequisites.json",
) -> Topics:
    assert topics or topics_file, "Either topics or topics_file must be provided."
    if topics_file:
        topics_file = Path(topics_file).with_suffix(".json")
        if topics_file.exists():
            print(f"Loading topics from {topics_file}")
            topics = Topics(**json.loads(topics_file.read_text()))
    assert topics, "Topics must be provided."

    prereq_system = f"""\
    {SYSTEM_PREFIX}
    You'll be given a question with a problem and solution and a list of other questions.
    Based on your knowledge of math, you have to decide which question form the list is a prerequisite to the given question.
    Just one question. If none are a prerequisite, or if the question is super easy for a highschooler, select 'None'.
    """
    prerequisites_file = Path(prerequisities_file).with_suffix(".json")
    prerequisites = (
        json.loads(prerequisites_file.read_text())
        if prerequisites_file.exists()
        else {}
    )
    question_values = list(topics.questions.values())

    for question_idx, question in enumerate(question_values):
        if question.id in prerequisites:
            continue
        prereq_qs = [
            prereq_q
            for prereq_q in question_values[:question_idx]
            if prereq_q.concept != question.concept
        ]
        prereq_strs = [
            f"<question>\n<id>\n{prereq_q.id}\n</id>\n{prereq_q.problem}\n{prereq_q.solution}\n</question>"
            for prereq_q in prereq_qs
        ]
        if not prereq_strs:
            continue
        candidate_questions = "\n\n".join(prereq_strs)
        messages = [
            user_message(
                f"<question>\n{question.problem}\n{question.solution}\n</question>\n\n<candidate_questions>\n{candidate_questions}\n</candidate_questions>"
            )
        ]
        try:
            prereq_id: str = ask_cld_or_oai(
                ask_cld=ask_cld,
                ask_oai=ask_oai,
                messages=messages,
                system=prereq_system,
                model=model,
                response_model=_t.Literal[
                    "None", *[question.id for question in prereq_qs]  # type: ignore
                ],
            )
            if prereq_id not in ["None", None]:
                topics.add_prerequisites(
                    id=question.id, prerequisites=topics.get(prereq_id)
                )
                if topics_file:
                    with open(Path(topics_file).with_suffix(".json"), "w") as f:
                        json.dump(topics.model_dump(), f, indent=2)
            prerequisites[question.id] = prereq_id
            with open(prerequisities_file, "w") as f:
                json.dump(prerequisites, f, indent=2)
            time.sleep(0.3)
        except Exception as e:
            print(e)
            continue
    return topics

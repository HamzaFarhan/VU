from collections import OrderedDict
from enum import Enum
from typing import Annotated, Literal, Sequence, Union

from pydantic import AfterValidator, BaseModel, Field

MIN_SUBTOPICS = 2
MAX_SUBTOPICS = 4
MIN_CONCEPTS = 2
MAX_CONCEPTS = 4
STR_ATTRS = ["topic", "subtopic", "concept", "question_number"]


def dict_to_ordereddict(d: dict | OrderedDict) -> OrderedDict:
    return OrderedDict(d)


def dict_to_xml(d: dict) -> str:
    xml_str = ""
    for key, value in d.items():
        xml_str += f"<{key}>\n{value}\n</{key}>\n"
    return xml_str.strip()


class CreatedSubtopic(BaseModel):
    name: Annotated[str, AfterValidator(str.title)]
    concepts: Annotated[
        list[str], AfterValidator(lambda concepts: [x.title() for x in concepts])
    ] = Field(
        f"{MIN_CONCEPTS}-{MAX_CONCEPTS} concepts covered in the subtopic.",
        min_length=MIN_CONCEPTS,
        max_length=MAX_CONCEPTS,
    )


class CreatedTopic(BaseModel):
    name: Annotated[str, AfterValidator(str.title)]
    subtopics: list[CreatedSubtopic] = Field(
        f"{MIN_SUBTOPICS}-{MAX_SUBTOPICS} ordered subtopics with concepts.",
        min_length=MIN_SUBTOPICS,
        max_length=MAX_SUBTOPICS,
    )


class Node(BaseModel):
    topic: str
    prerequisite_ids: list[str] = Field(default_factory=list)
    postrequisite_ids: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        id = ""
        for i, attr in enumerate(STR_ATTRS):
            if hasattr(self, attr):
                if i == 0:
                    id += getattr(self, attr)
                else:
                    id += f"_{getattr(self, attr)}"
        return id

    def __str__(self) -> str:
        node_str = f"<id>\n{self.id}\n</id>"
        for attr in STR_ATTRS:
            if hasattr(self, attr):
                # node_str += f"\n\n<{attr}>\n{getattr(self, attr)}\n</{attr}>"
                node_str += f"\n\n{dict_to_xml({attr: getattr(self, attr)})}"
        return node_str.strip()

    def add_prerequisite_id(self, id: str):
        if id == self.id:
            return
        prerequisite_ids = set(self.prerequisite_ids)
        prerequisite_ids.add(id)
        self.prerequisite_ids = list(prerequisite_ids)

    def add_postrequisite_id(self, id: str):
        if id == self.id:
            return
        postrequisite_ids = set(self.postrequisite_ids)
        postrequisite_ids.add(id)
        self.postrequisite_ids = list(postrequisite_ids)


class Topic(Node):
    # subtopics: dict[str, "Subtopic"] = Field(default_factory=OrderedDict)
    subtopics: Annotated[dict[str, "Subtopic"], AfterValidator(dict_to_ordereddict)] = (
        Field(default_factory=OrderedDict)
    )

    def get(self, id: str) -> Node | None:
        # if id.count("_") == 0 and id.count(".") > 0:
        # return self.get_numbered(id)
        split_id = id.split("_")
        if split_id[0] != self.id:
            print(f"ID {id} does not match topic {self.id}")
            return
        if len(split_id) == 1:
            return self
        elif len(split_id) == 2:
            return self.subtopics[id]
        elif len(split_id) == 3:
            return self.subtopics["_".join(split_id[:2])].concepts[id]
        elif len(split_id) == 4:
            return (
                self.subtopics["_".join(split_id[:2])]
                .concepts["_".join(split_id[:3])]
                .questions[id]
            )
        print(f"ID {id} not found")
        return None

    def add_subtopics(
        self,
        subtopics: Union[list[Union["Subtopic", str]], "Subtopic", str, None] = None,
    ):
        if subtopics is None or subtopics == [] or subtopics == "":
            return
        if not isinstance(subtopics, list):
            subtopics = [subtopics]
        for subtopic in subtopics:
            if isinstance(subtopic, str):
                subtopic = Subtopic(topic=self.topic, subtopic=subtopic)
            subtopic.topic = self.topic
            self.subtopics[subtopic.id] = subtopic

    def add_concepts(self, concepts: Union[list["Concept"], "Concept", None] = None):
        if concepts is None or concepts == []:
            return
        if not isinstance(concepts, list):
            concepts = [concepts]
        for concept in concepts:
            concept.topic = self.topic
            subtopic = self.subtopics.get(
                f"{concept.topic}_{concept.subtopic}",
                Subtopic(topic=concept.topic, subtopic=concept.subtopic),
            )
            subtopic.add_concepts(concept)
            self.subtopics[subtopic.id] = subtopic

    def add_questions(
        self, questions: Union[list["Question"], "Question", None] = None
    ):
        if questions is None or questions == []:
            return
        if not isinstance(questions, list):
            questions = [questions]
        for question in questions:
            question.topic = self.topic
            subtopic = self.subtopics.get(
                f"{question.topic}_{question.subtopic}",
                Subtopic(topic=question.topic, subtopic=question.subtopic),
            )
            subtopic.add_qestions(question)
            self.subtopics[subtopic.id] = subtopic

    def add_dependancies(
        self,
        id: str,
        dependancies: list[Node | str] | Node | str,
        mode: Literal["pre", "post"] = "pre",
    ):
        if dependancies is None or dependancies == [] or dependancies == "":
            return
        split_id = id.split("_")
        if split_id[0] != self.id:
            print(f"ID {id} does not match topic {self.id}")
            return
        if not isinstance(dependancies, list):
            dependancies = [dependancies]
        for dependancy in dependancies:
            if isinstance(dependancy, str):
                dependancy_id = dependancy
            else:
                dependancy_id = dependancy.id
            if dependancy_id == id:
                continue
            split_dependancy_id = dependancy_id.split("_")
            for i, splits in enumerate(zip(split_id, split_dependancy_id)):
                _, s2 = splits
                if i == 0:
                    if mode == "pre":
                        self.add_prerequisite_id(s2)
                    elif mode == "post":
                        self.add_postrequisite_id(s2)
                elif i == 1:
                    subtopic_id = "_".join(split_id[:2])
                    subtopic_dependancy_id = "_".join(split_dependancy_id[:2])
                    if mode == "pre":
                        self.subtopics[subtopic_id].add_prerequisite_id(
                            subtopic_dependancy_id
                        )
                    elif mode == "post":
                        self.subtopics[subtopic_id].add_postrequisite_id(
                            subtopic_dependancy_id
                        )
                elif i == 2:
                    subtopic_id = "_".join(split_id[:2])
                    concept_id = "_".join(split_id[:3])
                    concept_dependancy_id = "_".join(split_dependancy_id[:3])
                    if mode == "pre":
                        self.subtopics[subtopic_id].concepts[
                            concept_id
                        ].add_prerequisite_id(concept_dependancy_id)
                    elif mode == "post":
                        self.subtopics[subtopic_id].concepts[
                            concept_id
                        ].add_postrequisite_id(concept_dependancy_id)
                elif i == 3:
                    subtopic_id = "_".join(split_id[:2])
                    concept_id = "_".join(split_id[:3])
                    question_id = "_".join(split_id[:4])
                    question_dependancy_id = "_".join(split_dependancy_id[:4])
                    if mode == "pre":
                        self.subtopics[subtopic_id].concepts[concept_id].questions[
                            question_id
                        ].add_prerequisite_id(question_dependancy_id)
                    elif mode == "post":
                        self.subtopics[subtopic_id].concepts[concept_id].questions[
                            question_id
                        ].add_postrequisite_id(question_dependancy_id)


class Topics(BaseModel):
    topics: Annotated[dict[str, Topic], AfterValidator(dict_to_ordereddict)] = Field(
        default_factory=OrderedDict
    )

    @property
    def subtopics(self) -> dict[str, "Subtopic"]:
        subtopics = OrderedDict()
        for topic in self.topics.values():
            subtopics.update(topic.subtopics)
        return subtopics

    @property
    def concepts(self) -> dict[str, "Concept"]:
        concepts = OrderedDict()
        for subtopic in self.subtopics.values():
            concepts.update(subtopic.concepts)
        return concepts

    @property
    def questions(self) -> dict[str, "Question"]:
        questions = OrderedDict()
        for concept in self.concepts.values():
            questions.update(concept.questions)
        return questions

    def get_numbered(self, id: str) -> Node | None:
        split_id = id.split(".")
        if len(split_id) == 1:
            try:
                return list(self.topics.values())[int(split_id[0]) - 1]
            except IndexError:
                print(f"No topic found with ID {split_id[0]}")
                return
        elif len(split_id) == 2:
            try:
                topic = list(self.topics.values())[int(split_id[0]) - 1]
                return list(topic.subtopics.values())[int(split_id[1]) - 1]
            except IndexError:
                print(f"No subtopic found with ID {split_id[1]}")
                return
        elif len(split_id) == 3:
            try:
                topic = list(self.topics.values())[int(split_id[0]) - 1]
                subtopic = list(topic.subtopics.values())[int(split_id[1]) - 1]
                return list(subtopic.concepts.values())[int(split_id[2]) - 1]
            except IndexError:
                print(f"No concept found with ID {split_id[2]}")
                return
        elif len(split_id) == 4:
            try:
                topic = list(self.topics.values())[int(split_id[0]) - 1]
                subtopic = list(topic.subtopics.values())[int(split_id[1]) - 1]
                concept = list(subtopic.concepts.values())[int(split_id[2]) - 1]
                return list(concept.questions.values())[int(split_id[3]) - 1]
            except IndexError:
                print(f"No question found with ID {split_id[3]}")
                return
        print(f"ID {id} not found")
        return None

    def get(self, id: str) -> Node | None:
        if id.count("_") == 0 and id.count(".") == 0:
            topic_id = id
        elif id.count("_") > 0:
            topic_id = id.split("_")[0]
        elif id.count(".") > 0:
            return self.get_numbered(id)
        topic = self.topics.get(topic_id)
        if topic is None:
            print(f"No topic found with ID {topic_id}")
            return
        return topic.get(id)

    def add_topics(
        self,
        topics: list[Topic | str] | Topic | str | None = None,
    ):
        if topics is None or topics == [] or topics == "":
            return
        if not isinstance(topics, list):
            topics = [topics]
        for topic in topics:
            if isinstance(topic, str):
                topic = Topic(topic=topic)
            self.topics[topic.id] = topic

    def add_subtopics(
        self,
        subtopics: Union[list["Subtopic"], "Subtopic", None] = None,
    ):
        if subtopics is None or subtopics == [] or subtopics == "":
            return
        if not isinstance(subtopics, list):
            subtopics = [subtopics]
        for subtopic in subtopics:
            topic_key = subtopic.topic
            topic = self.topics.get(topic_key, Topic(topic=topic_key))
            topic.add_subtopics(subtopic)
            self.topics[topic_key] = topic

    def add_concepts(self, concepts: Union[list["Concept"], "Concept", None] = None):
        if concepts is None or concepts == [] or concepts == "":
            return
        if not isinstance(concepts, list):
            concepts = [concepts]
        for concept in concepts:
            topic_key = concept.topic
            topic = self.topics.get(topic_key, Topic(topic=topic_key))
            topic.add_concepts(concept)
            self.topics[topic_key] = topic

    def add_questions(
        self,
        questions: Union[list["Question"], "Question", None] = None,
    ):
        if questions is None or questions == []:
            return
        if not isinstance(questions, list):
            questions = [questions]
        for question in questions:
            topic_key = question.topic
            topic = self.topics.get(topic_key, Topic(topic=topic_key))
            topic.add_questions(question)
            self.topics[topic_key] = topic

    def add_prerequisites(
        self, id: str, prerequisites: list[Node | str] | Node | str | None = None
    ):
        if prerequisites is None or prerequisites == [] or prerequisites == "":
            return
        topic_key = id.split("_")[0]
        self.topics[topic_key].add_dependancies(
            id=id, dependancies=prerequisites, mode="pre"
        )
        if not isinstance(prerequisites, list):
            prerequisites = [prerequisites]
        for prereq in prerequisites:
            if isinstance(prereq, str):
                prereq_id = prereq
            else:
                prereq_id = prereq.id
            if prereq_id == id:
                continue
            topic_key = prereq_id.split("_")[0]
            self.topics[topic_key].add_dependancies(
                id=prereq_id, dependancies=id, mode="post"
            )


class Subtopic(Node):
    subtopic: str
    concepts: Annotated[dict[str, "Concept"], AfterValidator(dict_to_ordereddict)] = (
        Field(default_factory=OrderedDict)
    )

    def add_concepts(
        self,
        concepts: Union[Sequence[Union["Concept", str]], "Concept", str, None] = None,
    ):
        if concepts is None or concepts == [] or concepts == "":
            return
        if not isinstance(concepts, Sequence):
            concepts = [concepts]
        for concept in concepts:
            if isinstance(concept, str):
                concept = Concept(
                    topic=self.topic, subtopic=self.subtopic, concept=concept
                )
            concept.topic = self.topic
            concept.subtopic = self.subtopic
            self.concepts[concept.id] = concept

    def add_qestions(
        self,
        questions: Union[list["Question"], "Question", None] = None,
    ):
        if questions is None or questions == []:
            return
        if not isinstance(questions, list):
            questions = [questions]
        for question in questions:
            question.topic = self.topic
            question.subtopic = self.subtopic
            concept = self.concepts.get(
                f"{question.topic}_{question.subtopic}_{question.concept}",
                Concept(
                    topic=question.topic,
                    subtopic=question.subtopic,
                    concept=question.concept,
                ),
            )
            concept.add_questions(question)
            self.concepts[concept.id] = concept


class Concept(Node):
    subtopic: str
    concept: str
    questions: Annotated[dict[str, "Question"], AfterValidator(dict_to_ordereddict)] = (
        Field(default_factory=OrderedDict)
    )

    def add_questions(
        self,
        questions: Union[
            list[Union["BaseQuestion", "Question"]], "BaseQuestion", "Question", None
        ] = None,
    ):
        if questions is None or questions == []:
            return
        if not isinstance(questions, list):
            questions = [questions]
        for question in questions:
            new_question = Question(
                topic=self.topic,
                subtopic=self.subtopic,
                concept=self.concept,
                problem=question.problem,
                solution=question.solution,
                question_number=len(self.questions) + 1,
            )
            if isinstance(question, Question):
                new_question.subquestions = question.subquestions
                new_question.difficulty = question.difficulty
                new_question.code = question.code

            self.questions[new_question.id] = new_question


class BaseQuestion(BaseModel):
    problem: str
    solution: str

    def __str__(self) -> str:
        # return f"<problem>\n{self.problem}\n</problem>\n\n<solution>\n{self.solution}\n</solution>"
        return f"{dict_to_xml({'problem': self.problem})}\n\n{dict_to_xml({'solution': self.solution})}"


class QuestionDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Question(Node, BaseQuestion):
    subtopic: str
    concept: str
    question_number: int = 1
    difficulty: QuestionDifficulty = QuestionDifficulty.EASY
    subquestions: list[BaseQuestion] = Field(default_factory=list)
    code: str = ""

    def __str__(self) -> str:
        question_str = super().__str__()
        for i, subquestion in enumerate(self.subquestions, start=1):
            question_str += f"\n\n{dict_to_xml({f'subquestion_{i}': subquestion})}"
        return question_str.strip()


"""
We will get a file of topics.json where each topic will have subtopics and each subtopic will have concepts and each concept will have questions.
Each question will have a problem, solution, subquestions, and code to generate the question and subquestions with random variables.
Each topic, subtopic, concept, and question will also have prerequisites and postrequisites.
The prerequisites and postrequisites will be assigned on the question level in the add_dependencies function and then automatically updated for the concept, subtopic, and topic levels.
So if we assign question 1 of concept 1 of subtopic 1 of topic 1 as a prerequisite for question 2 of concept 2 of subtopic 2 of topic 2, then concept 1 of subtopic 1 of topic 1 will be a prerequisite for concept 2 of subtopic 2 of topic 2, subtopic 1 of topic 1 will be a prerequisite for subtopic 2 of topic 2, and topic 1 will be a prerequisite for topic 2.
We will load the file as a Topics object and then everything can be accessed by either name or numbered notation.
So question 3 of concept 2 of subtopic 1 of topic 1 can be accessed with topics.get("<topic1_name>_<subtopic1_name>_<concept2_name>_3") or topics.get("1.1.2.3")
The same for concepts, subtopics, and topics. So just topics.get("2.3") means subtopic 3 of topic 2.
And we already have a subtopic, we can do subtopic.get("3.5") to get question 5 of concept 3 of that subtopic.
"""

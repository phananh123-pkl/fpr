from langgraph.graph import StateGraph, END
from typing import TypedDict
import re
import getpass
import os
os.environ["DEEPSEEK_API_KEY"] = "sk-185ccf2f63564c93bb45b5bee4c24d08"
from langchain_deepseek import ChatDeepSeek
llm = ChatDeepSeek(model="deepseek-chat")

class CVState(TypedDict):
    file_path: str
    cv_text: str
    parsed_data: dict
    evaluation: str
    advice: str
    decision: str
    user_has_question: str
    user_query: str
    mes: str
    already_ask: bool
    ask_more: bool

import fitz

def extract_pdf_text(file_path: str) -> str:
    cv_text=''
    with fitz.open(file_path) as doc:
        for page in doc:
            cv_text+= page.get_text()
    return cv_text

def receive_cv(state: CVState) -> CVState:
    if 'cv_text' in state:
        del state['file_path']
        return state
    file_path=state['file_path']
    cv_text= extract_pdf_text(file_path)
    return {**state, 'cv_text': cv_text}

def parse_cv(state: CVState) -> CVState:
    if 'parse_data' in state:
        return state
    if 'file_path' not in state or not state['file_path']:
        return state
    prompt = f"""
    Đây là nội dung CV của ứng viên. 
    Hãy trích xuất những thông tin quan trọng nhất(tên, liên hệ, học vấn, kinh nghiệm, năng lực,....)
    Nội dung CV:\n\n{state["cv_text"]}
    """
    response = llm.invoke(prompt)
    return {**state, "parsed_data": response.content}

def evaluate_cv(state: CVState) -> CVState:
    if 'evaluation' in state:
        return state
    prompt_1 = f"""
    Dưới đây là dữ liệu đã trích xuất từ CV:\n{state["parsed_data"]}
    Liệt kê điểm mạnh và điểm yếu (3 điểm mạnh, 3 điểm yếu), từ đó đưa ra 2 câu đánh giá xem ứng viên phù hợp làm Data Analyst không
    """
    response_1 = llm.invoke(prompt_1)
    print(response_1.content)
    return {**state, "evaluation": response_1.content}


def make_decision(state: CVState) -> CVState:
    if 'decision' in state:
        return state
    prompt_3 = f"""
    Dựa vào đánh giá sau:\n{state["evaluation"]}
    Nhà tuyển dụng nên mời phỏng vấn hay từ chối ứng viên này? Giải thích ngắn gọn.
    """
    response_3 = llm.invoke(prompt_3)
    print(response_3.content)
    return {**state, "decision": response_3.content}

def cont(state: dict):
    if state.get('user_query'):
        state['already_ask'] = True
        return 'qna'
    elif state.get('already_ask'):
        return END
    else:
        return END
    
def qna(state:dict):
    query= state['user_query']
    prompt_4 = f"""
    Dưới đây là dữ liệu đã trích xuất từ CV:\n{state["parsed_data"]}
    Câu hỏi: {query}, dựa vào dữ liệu đã trích xuất, hãy trả lời câu hỏi của ứng viên đang apply vào vị trí Data Analyst
    """
    ans=llm.invoke(prompt_4)
    state['mes'] = ans
    state['ask_more'] = False
    print(ans)
    return state

def cont_2(state: CVState):
    if state['ask_more'] == True:
        return 'continue'
    else:
        return 'stop'
    
from IPython.display import Image, display
from langgraph.graph import StateGraph

builder=StateGraph(CVState)
    
builder.add_node("ReceiveCV", receive_cv)
builder.add_node("ParseCV", parse_cv)
builder.add_node("EvaluateCV", evaluate_cv)
builder.add_node("MakeDecision", make_decision)
builder.add_node('QNA',qna)

builder.set_entry_point("ReceiveCV")
builder.add_edge("ReceiveCV", "ParseCV")
builder.add_edge("ParseCV", "EvaluateCV")
builder.add_edge("EvaluateCV", "MakeDecision")
builder.add_conditional_edges('MakeDecision', cont,{'qna': 'QNA', END:END})
builder.add_conditional_edges('QNA', cont_2,{'continue': 'QNA', 'stop': END} )
builder.add_edge('QNA', END)
graph=builder.compile()

def run_graph(file_path: str, user_query: str = "", ask_more: bool = False):
    inputs = {
        "file_path": file_path,
        "user_query": user_query,
        "already_ask": False,
        "ask_more": ask_more,
    }
    result = graph.invoke(inputs)
    return result
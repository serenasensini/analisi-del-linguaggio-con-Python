'''
FIXME NOT WORKING
ImportError: cannot import name 'ChatBot' from 'chatterbot' (C:\Users\ISC_DELL\PycharmProjects\analisi-del-linguaggio-con-Python\chatterbot\__init__.py)
'''
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.response_selection import get_first_response
from chatterbot.comparisons import levenshtein_distance

import logging

logging.basicConfig(level=logging.CRITICAL)


bot = ChatBot(
    "Chappie",
    storage_adapter = "chatterbot.storage.SQLStorageAdapter",
    database = "./db.sqlite3",
    logic_adapters = [
        "chatterbot.logic.BestMatch"
    ],
    statement_comparison_function = levenshtein_distance,
    response_selection_method = get_first_response
)


with open("conv.txt") as f:
    conversation = f.readlines()
    trainer = ListTrainer(bot)
    trainer.train(conversation)


while True:
    try:
        user_input = input("Tu: ")
        bot_response = bot.get_response(user_input)
        print("Chappie: ", bot_response)
    except(KeyboardInterrupt, EOFError, SystemExit):
        print("GoodBye!")
        break


"""
It's a wrapper of a wrapper.

We'll use this so we can do bash calls.
"""

import argparse
import json

import telebot


class Telegram:
    def __init__(self, api_json_path):
        """
        Initiates telegram API bot.
        (Quite useful if you want to get notified of any
        status changes in the event that you're afk with model
        training.)
        """
        with open(api_json_path) as f:
            api_q = json.load(f)
        self.bot = telebot.TeleBot(api_q["api_private_key"])
        self.bot_chatid = api_q["chat_id"]

        @self.bot.message_handler(commands=["hello"])
        def send_something(message):
            print(message)
            pass

    def print(self, msg):
        """
        Sends messages to telegram chat ID.
        """
        if self.bot:
            self.bot.send_message(self.bot_chatid, msg)


def get_args():
    parser = argparse.ArgumentParser(description="telegram.py")
    # data options
    parser.add_argument(
        "-config", default="telegram.json", help="filepath to json config."
    )

    parser.add_argument("-m", required=True, help="messge to send.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    p = Telegram(args.config)
    p.print(args.m)

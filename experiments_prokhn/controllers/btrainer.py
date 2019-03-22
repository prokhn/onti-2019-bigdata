import os
import time
import json
import random
import requests
import traceback
import hashlib
import threading
import subprocess
from vk_api.bot_longpoll import VkBotEventType
from vkchatbot import Controller, Update, VkBot
from vkchatbot.obj import Message, Keyboard, VkKeyboardColor
from vkchatbot.ext import ConversationHandler


# class ExampleBot:
#     def __init__(self, admin_id):
#         self.data = {}
#         self.admin_id = admin_id
#
#     def main(self, update: Update):
#         if update.obj.text == 'Помощь':
#             update.reply_text('You passed main and asked for a help')
#             update.user.change_path('help')
#         else:
#             update.reply_text('You passed main')
#             update.user.change_path('main/sub1')
#
#     def main_sub1(self, update: Update):
#         update.reply_text('You passed main/sub1')
#         update.user.change_path('main/sub2')
#
#     def main_sub2(self, update: Update):
#         update.reply_text('You passed main/sub2')
#         update.user.change_path('main/sub2/sub21')
#
#     def main_sub2_sub21(self, update: Update):
#         update.reply_text('You passed main/sub2/sub21')
#         update.user.change_page('main')
#
#     def send_and_edit(self, update: Update):
#         msg = Message('Hi there')
#         msg_id = update.reply(msg)
#         time.sleep(1)
#         msg_new = Message('Again its meee')
#         update.bot.edit_message(update.obj.peer_id, msg_id, msg_new)
#
#     def new_message(self, update: Update):
#         kb = Keyboard(('Primary', VkKeyboardColor.PRIMARY), '\n',
#                       ('Positive', VkKeyboardColor.POSITIVE), ('Negative', VkKeyboardColor.NEGATIVE), '\n',
#                       ('Default', VkKeyboardColor.DEFAULT))
#
#         photo = update.bot.vk_upload.photo_messages('space.jpg')[0]  # type: dict
#         photo_id = photo['id']
#
#         word_from_url = 'https://sample-videos.com/doc/Sample-doc-file-100kb.doc'
#         docname = f'doc_tmp_{str(random.randint(10000000, 99999999))}.doc'
#         with open(docname, 'wb') as doc_file:
#             doc_bytes = requests.get(word_from_url).content
#             doc_file.write(doc_bytes)
#         doc = update.bot.vk_upload.document_message(docname, title='Sample document', peer_id=update.obj.peer_id)[0]
#         os.remove(docname)
#
#         msg = Message('Hello, its entry message with attachment, forward message, keyboard and sticker',
#                       attachments=[('photo', -update.bot.id, photo_id),
#                                    ('doc', update.obj.peer_id, doc['id'])],
#                       forward_messages=[update.obj.id],
#                       sticker_id=102,
#                       keyboard=kb)
#
#         update.reply(msg)
#         # update.user.change_page('main')
#
#     def smart_uploader(self, update: Update):
#         msg_before = Message('Im uploading one file...')
#         msg_final = Message('Here is your file! :)')
#
#         update.reply(msg_before)
#
#         word_from_url = 'https://sample-videos.com/doc/Sample-doc-file-100kb.doc'
#         word_filename = 'Sample_doc.doc'
#         update.upload_add(attach_type='doc', from_url=word_from_url, doc_title='Sample from web 1')
#         update.upload_add(attach_type='doc', filename=word_filename, doc_title='Sample from tmp 1')
#         update.upload_add(attach_type='doc', from_url=word_from_url, doc_title='Sample from web 2')
#         update.upload_add(attach_type='doc', filename=word_filename, doc_title='Sample from tmp 2')
#         update.upload_add(attach_type='doc', from_url=word_from_url, doc_title='Sample from web 3')
#         update.upload_add(attach_type='doc', filename=word_filename, doc_title='Sample from tmp 3')
#         update.upload_add(attach_type='doc', from_url=word_from_url, doc_title='Sample from web 4')
#         update.upload_add(attach_type='doc', filename=word_filename, doc_title='Sample from tmp 4')
#         update.upload_start(msg_final, one_by_one=True)
#
#     def ask_help(self, update: Update):
#         update.reply_text('You passed help')
#         update.user.change_path('main')
#
#     def throwing_exc(self, update: Update):
#         update.reply_text('I will throw one exception, try to catch it :)')
#         raise ValueError('Hello, how are you today?')
#
#     def chat_tests(self, update: Update):
#         print('Chats:', update.bot.chats, '\nUsers:', update.bot.users)
#
#     def on_errors(self, update: Update, exc):
#         strsxc = traceback.format_exception(*exc)
#         del strsxc[1:2]
#         # print('Error!\n', *strsxc)
#         update.reply_text('Sorry, some error occured')
#         # admin_msg = Message(text=f'Error!\n{str(update)}\n=== Traceback info ===\n{"".join(strsxc)}')
#         # update.bot.send_message(self.admin_id, admin_msg)


class BotLogger:
    def __init__(self, admin_id):
        self.data = {}

        self.auth_keys = {'7cd93a59af53470951f5cefdc': True,
                          '03e8aa1c34b0d300d886cca63': True,
                          '1d996e033d612d9af2b44b700': True,
                          'fe52470716d7719f811d9571e': True}
        self.users = []

        self.sessions = []

        self.admin_id = admin_id

    def entry(self, update: Update):
        update.reply_text('Hello, send me access key to auth')
        update.user.change_path('auth')

    def auth(self, update: Update):
        key = hashlib.sha256(update.obj.text.encode('utf-8')).hexdigest()[:25]
        if key in self.auth_keys.keys():
            if self.auth_keys[key]:
                self.auth_keys[key] = False
                self.users.append(update.user.id)
                update.obj.text = ''
                return self.main(update)
            else:
                update.reply_text('Key expired. Try again')
        else:
            update.reply_text('Key is incorrect. Try again')

    def main(self, update: Update):
        if update.user.id not in self.users:
            update.reply_text('Something went wrong: you are not allowed to go here. Enter key again')
            update.user.change_page('auth')
            return

        if update.obj.text == '':
            update.reply_text('Correct key. Welcome to LoggerBot')
            update.user.change_page('main')
            return

        proc = subprocess.Popen(update.obj.text, stdout=subprocess.PIPE, shell=True)
        result = proc.stdout.read().decode('utf-8')
        update.reply_text(result)

    def on_errors(self, update: Update, exc):
        strsxc = traceback.format_exception(*exc)
        del strsxc[1:2]
        print('Error!\n', *strsxc)
        # update.reply_text('Sorry, some error occured')
        # admin_msg = Message(text=f'Error!\n{str(update)}\n=== Traceback info ===\n{"".join(strsxc)}')
        # update.bot.send_message(self.admin_id, admin_msg)


if __name__ == '__main__':
    """
    config.json ->
        {
            "group_id": Your group id,
            "access_token": Group access token,
            "admin_id": Your VK id (bot can send you exceptions)
        }
    """
    with open('config.json') as file:
        config = json.load(file)
        group_id = config['group_id']
        access_token = config['access_token']
        admin_id = config['admin_id']

    bot_controller = Controller(group_id, access_token)

    loggerBot = BotLogger(admin_id=admin_id)

    ch = ConversationHandler(entry_callback=loggerBot.entry, pages={
        'auth': loggerBot.auth,
        'main': loggerBot.main,
    })

    bot_controller.handlers[VkBotEventType.MESSAGE_NEW] = ch
    bot_controller.error_handler = loggerBot.on_errors

    # Config
    # If True - bot will send sticker first, then text, attachments and other
    bot_controller.bot.config['send_stickers_first'] = False

    bot_controller.run()

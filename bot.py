import sys
import os
import shutil

import aiohttp
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils.helper import Helper, HelperMode, ListItem

import interface


class AppStates(Helper):
    mode = HelperMode.snake_case

    STATE_LOAD_STYLE = ListItem()
    STATE_LOAD_CONTENT = ListItem()

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

bot = Bot(token=os.environ['TELEGRAM_TOKEN'])
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())



net_instance = None
def get_net():
    global net_instance
    if net_instance is None:
        net_instance = interface.create_net()
    return net_instance

async def do_style(style, content, image):
    net = get_net()
    interface.do_style(net, style, content, image)

@dp.message_handler(state='*', commands=['help'])
async def process_help_command(message: types.Message):
    n_styles = len(interface.style_images())
    msg = \
        'Я DLS Style transfer bot, который является выпускным проектом Малых Михаила в Deep Learning School.\n' + \
        'Я умею переносить разные стили на Ваши фотографии. Стили Вы можете задать сами или выбрать один из предложенных. ' + \
        'Чтобы посмотреть доступные стили, используйте команду "/styles n", где n от 0 до {} - номер стиля.\n'.format(n_styles - 1) + \
        'Чтобы начать и загрузить/выбрать стилевую картинку, наберите /go.\n' + \
        'Затем загрузите свою фотографию и Вы получите стилизованное картину в ответ!\n' + \
        'Вы всегда можете посмотреть эту информацию снова, набрав  команду /help.\n'
    
    await message.reply(msg, reply=False)

@dp.message_handler(state='*', commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет!\nЯ DLS Style transfer bot!\nНабери /go, чтобы начать, или /help, чтобы посмотреть справку.", reply=False)

@dp.message_handler(state='*', commands=['styles'])
async def process_styles_command(message: types.Message):
    args = message.get_args()
    styles = list(interface.examples_styles_images())
    n_styles = len(styles)
    if args == "":
        return await message.reply('Формат команды: "/styles n", где n от 0 до {}'.format(n_styles - 1), reply=False)
    if args == "all" and False:
        await types.ChatActions.upload_photo()
        media = types.MediaGroup()
        for st, im in styles:
            media.attach_photo(types.InputFile(st), 'Стиль')
            media.attach_photo(types.InputFile(im), 'Результат')
        return await message.reply_media_group(media=media, reply=False)

    if not is_int(args):
        return await message.reply('Формат команды: "/styles n", где n от 0 до {}'.format(n_styles - 1), reply=False)
    
    s = int(args)
    if s < 0 or s >= n_styles:
        return await message.reply('Номер стиля может быть только от от 0 до {}'.format(n_styles - 1), reply=False)
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    media.attach_photo(types.InputFile(interface.example_content()), 'Контент')
    media.attach_photo(types.InputFile(styles[s][0]), 'Стиль')
    media.attach_photo(types.InputFile(styles[s][1]), 'Результат')
    await message.reply_media_group(media=media, reply=False)

@dp.message_handler(state='*', commands=['go'])
async def process_go_command(message: types.Message):
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(AppStates.STATE_LOAD_STYLE[0])
    n_styles = len(interface.style_images())
    await message.reply('Начнём! Загрузи стилевую картинку или число от 0 до {}, чтобы выбрать готовый стиль.'.format(n_styles - 1), reply=False)

@dp.message_handler(state=AppStates.STATE_LOAD_STYLE, content_types=types.ContentType.ANY) 
async def state_go(message: types.Message):
    state = dp.current_state(user=message.from_user.id)

    if len(message.photo) == 0:
        if is_int(message.text):
            s = int(message.text)
            styles = interface.style_images()
            n_styles = len(styles)
            if s < 0 or s >= n_styles:
                return await message.reply('Номер стиля может быть только от 0 до {}.'.format(n_styles - 1), reply=False)
            dstyle = 'data/{}_style.jpg'.format(message.from_user.id)
            shutil.copy(styles[s], dstyle)

            await types.ChatActions.upload_photo()
            media = types.MediaGroup()
            media.attach_photo(types.InputFile(dstyle), 'Вы выбрали такой стиль.')
            await message.reply_media_group(media=media, reply=False)
        else:
            return await message.reply('Вы не загрузили стилевую картинку... Загрузите, пожалуйста.', reply=False)
    else:
        await message.photo[-1].download('data/{}_style.jpg'.format(message.from_user.id))
    await state.set_state(AppStates.STATE_LOAD_CONTENT[0])
    await message.reply('Загрузи контентную картинку.', reply=False)

@dp.message_handler(state=AppStates.STATE_LOAD_CONTENT, content_types=types.ContentType.ANY)
async def state_style(message: types.Message):
    state = dp.current_state(user=message.from_user.id)

    if len(message.photo) == 0:
        return await message.reply('Вы не загрузили контентную картинку... Загрузите, пожалуйста.', reply=False)

    path_content = 'data/{}_content.jpg'.format(message.from_user.id)
    path_style = 'data/{}_style.jpg'.format(message.from_user.id)
    path_output = 'data/{}_output.jpg'.format(message.from_user.id)
    
    message.photo[-1].download(path_content)
    await do_style(path_style, path_content, path_output)

    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    media.attach_photo(types.InputFile(path_output), 'Вот результат!')

    await message.reply_media_group(media=media, reply=False)
    await state.reset_state()
    await message.reply("Набери /go, чтобы начать снова.", reply=False)



@dp.message_handler(state='*')
async def echo_message(msg: types.Message):
    await bot.send_message(msg.from_user.id, 'Набери /go, чтобы начать, или /help, чтобы посмотреть справку.')

async def shutdown(dispatcher: Dispatcher):
    await dispatcher.storage.close()
    await dispatcher.storage.wait_closed()

if __name__ == '__main__':
    executor.start_polling(dp, on_shutdown=shutdown)
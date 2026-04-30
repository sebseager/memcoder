# fogleman__Craft easy ledger

Source commit: `d6888a6e1e54340358ce25b1220f410541360b6b`

## Overview / Purpose

- [overview_purpose_1] Overview / Purpose (easy) - Explains Craft's purpose as a small cross-platform Minecraft-like game, how the main client organizes world state and rendering, and how configuration, persistence, multiplayer, and build setup fit together. Example questions: What kind of game is Craft and what major features does its README emphasize?; How does src/main.c organize chunks, players, input, networking, and rendering at a high level?; What initialization work happens before Craft enters its main game loop?; Which defaults and feature toggles does src/config.h provide for the Craft client?; How does the CMake build assemble the Craft executable and its dependencies?

## Setup, Build, and Run

- [setup_build_run_1] Setup, Build, and Run (easy) - Explains how Craft is built from source, how the client and multiplayer server are launched, and how the Python server depends on the compiled world-generation library. Example questions: What source dependencies does Craft expect before running the normal CMake and make build?; How do I start the Craft client locally versus connect it to a multiplayer host?; What extra GCC-built artifact does the Python Craft server need for terrain generation?; How does the server cleanup mode differ from starting the Craft TCP service?; Where does builder.py fit into Craft's setup and server workflow?

## Gameplay, Controls, and Commands

- [gameplay_controls_commands_1] Gameplay, Controls, and Commands (easy) - Explains how Craft turns player input into movement, view changes, block editing, signs, chat, slash commands, hit testing, and collision behavior. Example questions: How does Craft switch between normal gameplay controls and chat, command, or sign typing?; What do the mouse buttons do for destroying blocks, creating blocks, toggling lights, and picking block types?; How does backquote sign entry decide which block face receives the text?; How does handle_movement combine walking, flying, jumping, zoom, and arrow-key camera input?; Which kinds of local builder commands can Craft parse from slash commands?

## World Generation and Chunks

- [world_generation_chunks_1] World Generation and Chunks (easy) - Explains how Craft generates deterministic terrain, stores block data by chunk, loads and updates chunks around the player, and builds renderable chunk buffers. Example questions: How does Craft use Simplex noise to build the default terrain in each chunk?; What role do Craft's chunk create, render, sign, and delete radii play around the player?; How are generated terrain and sqlite block deltas combined when a chunk loads?; What does Craft's Map structure store for a chunk's block data?; How does render_chunks decide which loaded chunks are actually drawn?

## Rendering and Graphics Pipeline

- [rendering_graphics_pipeline_1] Rendering and Graphics Pipeline (easy) - Explains how Craft builds chunk, block, plant, sign, sky, text, and player geometry into OpenGL buffers, then renders them with shaders, culling, daylight, fog, transparency, and ambient occlusion. Example questions: How does Craft organize its OpenGL shader programs and texture units for blocks, text, signs, and the sky?; What does Craft store in block and plant vertex buffers before drawing them?; How does the rendering path decide which chunks are visible enough to draw?; How are magenta texture pixels handled for glass, plants, and other cutout geometry?; What role do the matrix helpers play when rendering the world, HUD, item preview, and picture-in-picture view?

## Persistence, Database, and Cache

- [persistence_database_cache_1] Persistence, Database, and Cache (easy) - Explains how Craft persists world edits, player state, signs, lights, auth tokens, and multiplayer cache keys through sqlite and a background write queue. Example questions: What world data does Craft store in sqlite instead of regenerating every session?; How does db_init set up Craft's persistence layer before gameplay starts?; Why do Craft's block and light database writes go through a background worker?; How do the /online and /offline commands choose which sqlite database file Craft uses?

## Multiplayer Protocol and Server

- [multiplayer_protocol_server_1] Multiplayer Protocol and Server (easy) - Explains Craft's socket-based multiplayer protocol, how the C client enters online mode, how chunk cache keys and world updates flow through the Python server, and how login, chat, players, time, lights, and signs are synchronized. Example questions: How does Craft's line-based multiplayer protocol move block, light, sign, chat, and position updates between client and server?; What does client.c do with incoming socket data before main.c parses it?; How does server.py validate and broadcast multiplayer block edits?; What happens during Craft's multiplayer login flow when a saved identity token is present?; How does parse_buffer update local players, chat, chunk redraws, and server time?

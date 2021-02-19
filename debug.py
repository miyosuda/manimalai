from envs.arena_config import ArenaConfig

config = ArenaConfig("./configurations/1-1-1.yml")

arena = config.arenas[0]
print(arena.t)

for item in arena.items:
    print(item.name)
    print(item.positions)
    
    if item.name == 'Agent':
        
    

fiber = require('fiber')

local index_count = tonumber(arg[1])
local fiber_count = tonumber(arg[2])
local REPLACE_PER_TXN_COUNT = tonumber(arg[3]) or 2
local divisor = tonumber(arg[4]) or 3

os.execute('rm -rf *.snap *.xlog')

box.cfg{memtx_use_mvcc_engine=true}

box.schema.space.create("test")

for i = 1, index_count do
	box.space.test:create_index(
		('idx_%d'):format(i),
		{parts={{i}}, unique=true}
	)
end

local fun = require('fun')
local tuples = {}

for i = 1, fiber_count do
    tuples[i] = fun.totable(fun.take(index_count, fun.duplicate(i)))
end

local fibers = {}
local conds = {}
for i = 1, fiber_count do
	conds[i] = fiber.cond()
end
for i = 1, fiber_count do
	-- each chain has length REPLACE_PER_TXN_COUNT * X,
	-- where X ~ Bin(fiber_count, divisor/fiber_count)
	-- and E[X] = divisor and Var[X] = divisor(1 - divisor/fiber_count)
	local value = math.random(1, math.floor(fiber_count / divisor))
	local tuple = tuples[value] -- {value, value, ..., value}
	fibers[i] = fiber.create(function()
		box.begin()
			for _ = 1, REPLACE_PER_TXN_COUNT do
				box.space.test:replace(tuple)
			end
			conds[i]:wait()
		box.commit()
	end)
	fibers[i]:set_joinable(true)
end

for i = 1, fiber_count do
	conds[i]:signal()
	local ok, err = fibers[i]:join()
	assert(ok and err == nil)
end

os.exit()

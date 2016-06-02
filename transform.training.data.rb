File.open('states.training.transformed.csv', 'w') do |f|
    File.read('states.training.csv').lines.shuffle.each do |line|
        parts = line.strip.split(',')
        label = parts[-1]
        state = parts[0..-2].map { |p| p.split(';') }.flatten
        turn = state.select { |cell| cell == '0' }.size % 2 == 0 ? 1 : 2

        expand = [turn == 1 ? 0 : 1] + state.dup.map { |cell| cell == '2' ? '0' : cell } + state.dup.map { |cell| cell == '1' ? '0' : (cell == '2' ? '1' : '0') }

        f << "#{expand.join(',')},#{label}\n"
    end
end
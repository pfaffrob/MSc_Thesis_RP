-- Lua filter to automatically number H1 headers with "Chapter" prefix
local chapter_count = 0

function Header(el)
  -- Only process H1 headers (level 1)
  if el.level == 1 then
    -- Check if the header has the "unnumbered" class
    if not el.classes:includes("unnumbered") then
      chapter_count = chapter_count + 1
      -- Prepend "Chapter N" to the header content
      table.insert(el.content, 1, pandoc.Str("Chapter " .. chapter_count .. " "))
    end
  end
  return el
end

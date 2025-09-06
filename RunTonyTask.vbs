Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
base = FSO.GetParentFolderName(WScript.ScriptFullName)
cmd = "cmd /c """ & base & "\RunTonyTask.bat""" 
WshShell.Run cmd, 0, False


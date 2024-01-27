(Get-ACL -Path ".").Access | Format-Table IdentityReference,FileSystemRights,AccessControlType,IsInherited,InheritanceFlags -AutoSize

$ACL = Get-ACL -Path "."
$AccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Users","FullControl","Allow")
$ACL.SetAccessRule($AccessRule)
$ACL | Set-Acl -Path "Test1.txt"
(Get-ACL -Path "Test1.txt").Access | Format-Table IdentityReference,FileSystemRights,AccessControlType,IsInherited,InheritanceFlags -AutoSize

